"""
SEC-BERT Qualitative Feature Extractor v2 (Full-Section & Adverse-Risk Evidence)

Key improvements over v1:
1. Extended section representation: processes up to 8 chunks for Risk Factors (4,096 tokens, double v1)
   and 4 chunks for MD&A, with optimal CPU cache mini-batching.
2. Robust section extraction: multi-pattern fallback regex to capture Risk Factors and MD&A in filings where v1 failed.
3. Directional adverse-risk evidence score: computes projection of YoY semantic delta (E_t - E_{t-1})
   along a curated SEC-BERT adverse disclosure vector (litigation, regulatory sanctions, impairment, liquidity distress, covenant breach).
4. Disclosure novelty score: chunk-level novelty ratio against prior filing chunks.
5. Pretraining overlap flag: tags filings prior to 2020 as overlapping with SEC-BERT pretraining data.
6. Persisted into a separate versioned table: `qualitative_transformer_features_v2`.
"""

from __future__ import annotations
import os
import sys
import time
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
import mysql.connector

from config.settings import get_settings
from src.qualitative.html_cleaner import clean_html, preprocess_text
from src.qualitative.section_extractor import extract_risk_factors, extract_mda
from src.utils.logging import get_logger

logger = get_logger("secbert_extractor_v2")

MODEL_CHECKPOINT = "nlpaueb/sec-bert-base"
CHUNK_SIZE = 512
MAX_RF_CHUNKS = 8
MAX_MDA_CHUNKS = 4
SUB_BATCH_SIZE = 4

torch.set_num_threads(8)

ADVERSE_REFERENCE_TEXT = (
    "Material adverse effect, significant financial impairment, liquidity distress, "
    "covenant default, formal regulatory investigation, severe litigation sanctions, "
    "cybersecurity breach, material weakness in internal control, substantial decline in customer demand."
)


def setup_transformer_table_v2(cursor):
    """Creates the qualitative_transformer_features_v2 table if it doesn't exist."""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qualitative_transformer_features_v2 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        ticker VARCHAR(10) NOT NULL,
        cik VARCHAR(20),
        fiscal_year INT,
        form VARCHAR(10),
        filing_date DATE NOT NULL,
        report_date DATE,
        accession_number VARCHAR(30) NOT NULL UNIQUE,
        prior_accession_number VARCHAR(30),
        prior_filing_date DATE,
        extraction_status VARCHAR(20),
        pretraining_overlap_flag TINYINT DEFAULT 0,
        secbert_v2_rf_available TINYINT DEFAULT 0,
        secbert_v2_mda_available TINYINT DEFAULT 0,
        secbert_v2_has_prior TINYINT DEFAULT 0,
        secbert_v2_rf_chunks INT DEFAULT 0,
        secbert_v2_rf_tokens INT DEFAULT 0,
        secbert_v2_mda_chunks INT DEFAULT 0,
        secbert_v2_rf_cosine_sim FLOAT DEFAULT 1.0,
        secbert_v2_rf_semantic_drift FLOAT DEFAULT 0.0,
        secbert_v2_rf_novelty_score FLOAT DEFAULT 0.0,
        secbert_v2_adverse_evidence_score FLOAT DEFAULT 0.0,
        secbert_v2_rf_norm_diff FLOAT DEFAULT 0.0,
        secbert_v2_mda_cosine_sim FLOAT DEFAULT 1.0,
        secbert_v2_mda_semantic_drift FLOAT DEFAULT 0.0,
        secbert_v2_rf_mda_divergence FLOAT DEFAULT 0.0,
        model_name VARCHAR(50) DEFAULT 'nlpaueb/sec-bert-base',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_ticker_date (ticker, filing_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)


class SecBertFeatureExtractorV2:
    """Full-section SEC-BERT feature extractor with adverse risk projection."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.settings = get_settings()
        logger.info(f"Loading frozen {MODEL_CHECKPOINT} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.model = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(self.device)
        self.model.eval()

        # Precompute Adverse Reference Vector
        ref_tokens = self.tokenizer(ADVERSE_REFERENCE_TEXT, return_tensors="pt", max_length=CHUNK_SIZE, truncation=True)
        with torch.no_grad():
            ref_out = self.model(input_ids=ref_tokens["input_ids"].to(self.device), attention_mask=ref_tokens["attention_mask"].to(self.device))
            ref_hidden = ref_out.last_hidden_state
            mask = ref_tokens["attention_mask"].unsqueeze(-1).expand(ref_hidden.size()).float()
            ref_emb = (torch.sum(ref_hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)).squeeze(0).cpu().numpy()
            self.adverse_ref_vector = ref_emb / np.linalg.norm(ref_emb)

    def robust_extract_sections(self, ticker: str, accession: str, cleaned_text: str) -> Tuple[str, str, str]:
        """Extracts Item 1A and Item 7 with regex fallback patterns."""
        rf_res = extract_risk_factors(ticker, accession, cleaned_text)
        mda_res = extract_mda(ticker, accession, cleaned_text)

        rf_text = rf_res.raw_text
        mda_text = mda_res.raw_text

        # Fallback for Item 1A if primary regex returned empty
        if not rf_text or len(rf_text) < 200:
            pattern = re.compile(r"(?:item\s*1a[\.\:\-\s]+risk\s*factors)(.*?)(?:item\s*(?:1b|2|3|7)[\.\:\-\s])", re.IGNORECASE | re.DOTALL)
            m = pattern.search(cleaned_text)
            if m:
                rf_text = m.group(1).strip()

        # Fallback for Item 7 if primary regex returned empty
        if not mda_text or len(mda_text) < 200:
            pattern = re.compile(r"(?:item\s*7[\.\:\-\s]+management['\’]?s\s*discussion)(.*?)(?:item\s*(?:7a|8)[\.\:\-\s])", re.IGNORECASE | re.DOTALL)
            m = pattern.search(cleaned_text)
            if m:
                mda_text = m.group(1).strip()

        status = "SUCCESS" if (rf_text and mda_text) else ("PARTIAL" if (rf_text or mda_text) else "FAILED")
        return rf_text, mda_text, status

    def embed_section_chunks(self, text: str, max_chunks: int) -> Tuple[Optional[np.ndarray], List[np.ndarray], int, int]:
        """Processes chunks with CPU cache-friendly sub-batching."""
        if not text or not text.strip():
            return None, [], 0, 0

        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        total_tokens = len(input_ids)

        chunks = [input_ids[i:i+CHUNK_SIZE] for i in range(0, len(input_ids), CHUNK_SIZE)]
        selected_chunks = chunks[:max_chunks]
        if not selected_chunks:
            return None, [], 0, 0

        chunk_embeddings = []
        with torch.no_grad():
            for start_idx in range(0, len(selected_chunks), SUB_BATCH_SIZE):
                sub_batch = selected_chunks[start_idx:start_idx+SUB_BATCH_SIZE]
                b_size = len(sub_batch)
                padded_batch = torch.full((b_size, CHUNK_SIZE), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
                for idx, c in enumerate(sub_batch):
                    padded_batch[idx, :len(c)] = c.to(self.device)
                attn_mask = (padded_batch != self.tokenizer.pad_token_id).long()

                out = self.model(input_ids=padded_batch, attention_mask=attn_mask)
                last_hidden = out.last_hidden_state
                mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                batch_emb = (sum_embeddings / sum_mask).cpu().numpy()
                for e in batch_emb:
                    chunk_embeddings.append(e)

        chunk_list = [c / max(1e-9, np.linalg.norm(c)) for c in chunk_embeddings]
        section_emb = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(section_emb)
        if norm > 0:
            section_emb = section_emb / norm
        return section_emb, chunk_list, len(selected_chunks), total_tokens

    def extract_all_filings(self):
        """Processes all 168 filings in MongoDB and populates qualitative_transformer_features_v2."""
        mongo = MongoClient(self.settings.mongo.uri)
        db = mongo[self.settings.mongo.database]
        coll = db["sec_filings"]

        mysql_conn = mysql.connector.connect(
            host=self.settings.mysql.host,
            port=self.settings.mysql.port,
            user=self.settings.mysql.user,
            password=self.settings.mysql.password,
            database=self.settings.mysql.database
        )
        cursor = mysql_conn.cursor(dictionary=True)
        setup_transformer_table_v2(cursor)
        mysql_conn.commit()

        cursor.execute("SELECT accession_number FROM qualitative_transformer_features_v2;")
        already_processed = set(r["accession_number"] for r in cursor.fetchall())
        logger.info(f"Already processed v2 filings: {len(already_processed)}")

        docs = list(coll.find({}, {
            "ticker": 1, "cik": 1, "fiscal_year": 1, "filing_type": 1,
            "filing_date": 1, "report_date": 1, "accession_number": 1, "raw_content": 1
        }))
        logger.info(f"Loaded {len(docs)} documents from MongoDB for v2 extraction")

        ticker_docs: Dict[str, List[Dict[str, Any]]] = {}
        for d in docs:
            ticker_docs.setdefault(d["ticker"], []).append(d)

        for t in ticker_docs:
            ticker_docs[t].sort(key=lambda x: (pd.to_datetime(x["filing_date"]), x["accession_number"]))

        total_saved = len(already_processed)
        t_start = time.time()

        for ticker, t_list in ticker_docs.items():
            prior_rf_emb: Optional[np.ndarray] = None
            prior_rf_chunks: List[np.ndarray] = []
            prior_mda_emb: Optional[np.ndarray] = None
            prior_meta: Optional[Dict[str, Any]] = None

            for d in t_list:
                acc = d["accession_number"]
                f_date = pd.to_datetime(d["filing_date"]).strftime("%Y-%m-%d")
                r_date = pd.to_datetime(d.get("report_date")).strftime("%Y-%m-%d") if d.get("report_date") else None
                fy = d.get("fiscal_year")
                form = d.get("filing_type", "10-K")
                cik = d.get("cik")

                # If already processed, update prior reference and continue
                if acc in already_processed:
                    prior_meta = {"accession_number": acc, "filing_date": f_date}
                    continue

                pretraining_overlap = 1 if pd.to_datetime(f_date).year <= 2019 else 0
                raw_html = d.get("raw_content", "")
                cleaned = preprocess_text(clean_html(raw_html))
                rf_text, mda_text, status = self.robust_extract_sections(ticker, acc, cleaned)

                rf_emb, rf_chunk_list, rf_chunks, rf_tokens = self.embed_section_chunks(rf_text, MAX_RF_CHUNKS)
                mda_emb, mda_chunk_list, mda_chunks, mda_tokens = self.embed_section_chunks(mda_text, MAX_MDA_CHUNKS)

                has_prior = 1 if prior_meta is not None else 0
                prior_acc = prior_meta["accession_number"] if prior_meta else None
                prior_fdate = prior_meta["filing_date"] if prior_meta else None

                rf_available = 1 if rf_emb is not None else 0
                mda_available = 1 if mda_emb is not None else 0

                # 1. Unsigned Semantic Drift
                rf_cosine = 1.0
                rf_drift = 0.0
                rf_norm_diff = 0.0
                novelty_score = 0.0
                adverse_evidence = 0.0

                if rf_emb is not None and prior_rf_emb is not None:
                    sim = float(np.dot(rf_emb, prior_rf_emb))
                    rf_cosine = float(np.clip(sim, -1.0, 1.0))
                    rf_drift = float(1.0 - rf_cosine)
                    rf_norm_diff = float(np.linalg.norm(rf_emb - prior_rf_emb))

                    # 2. Chunk-level Novelty Score
                    if rf_chunk_list and prior_rf_chunks:
                        novel_chunks = 0
                        for cur_c in rf_chunk_list:
                            max_sim_to_prior = max(float(np.dot(cur_c, p_c)) for p_c in prior_rf_chunks)
                            if max_sim_to_prior < 0.88:
                                novel_chunks += 1
                        novelty_score = float(novel_chunks / len(rf_chunk_list))

                    # 3. Directional Adverse Risk Evidence Projection
                    delta_vec = rf_emb - prior_rf_emb
                    delta_norm = np.linalg.norm(delta_vec)
                    if delta_norm > 1e-6:
                        norm_delta = delta_vec / delta_norm
                        adverse_evidence = float(np.dot(norm_delta, self.adverse_ref_vector))

                # MD&A drift
                mda_cosine = 1.0
                mda_drift = 0.0
                if mda_emb is not None and prior_mda_emb is not None:
                    sim = float(np.dot(mda_emb, prior_mda_emb))
                    mda_cosine = float(np.clip(sim, -1.0, 1.0))
                    mda_drift = float(1.0 - mda_cosine)

                # RF vs MD&A Divergence
                rf_mda_div = 0.0
                if rf_emb is not None and mda_emb is not None:
                    rf_mda_div = float(1.0 - np.clip(np.dot(rf_emb, mda_emb), -1.0, 1.0))

                insert_sql = """
                INSERT INTO qualitative_transformer_features_v2 (
                    ticker, cik, fiscal_year, form, filing_date, report_date,
                    accession_number, prior_accession_number, prior_filing_date,
                    extraction_status, pretraining_overlap_flag,
                    secbert_v2_rf_available, secbert_v2_mda_available,
                    secbert_v2_has_prior, secbert_v2_rf_chunks, secbert_v2_rf_tokens,
                    secbert_v2_mda_chunks, secbert_v2_rf_cosine_sim,
                    secbert_v2_rf_semantic_drift, secbert_v2_rf_novelty_score,
                    secbert_v2_adverse_evidence_score, secbert_v2_rf_norm_diff,
                    secbert_v2_mda_cosine_sim, secbert_v2_mda_semantic_drift,
                    secbert_v2_rf_mda_divergence, model_name
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) ON DUPLICATE KEY UPDATE
                    prior_accession_number = VALUES(prior_accession_number),
                    prior_filing_date = VALUES(prior_filing_date),
                    extraction_status = VALUES(extraction_status),
                    pretraining_overlap_flag = VALUES(pretraining_overlap_flag),
                    secbert_v2_rf_available = VALUES(secbert_v2_rf_available),
                    secbert_v2_mda_available = VALUES(secbert_v2_mda_available),
                    secbert_v2_has_prior = VALUES(secbert_v2_has_prior),
                    secbert_v2_rf_chunks = VALUES(secbert_v2_rf_chunks),
                    secbert_v2_rf_tokens = VALUES(secbert_v2_rf_tokens),
                    secbert_v2_mda_chunks = VALUES(secbert_v2_mda_chunks),
                    secbert_v2_rf_cosine_sim = VALUES(secbert_v2_rf_cosine_sim),
                    secbert_v2_rf_semantic_drift = VALUES(secbert_v2_rf_semantic_drift),
                    secbert_v2_rf_novelty_score = VALUES(secbert_v2_rf_novelty_score),
                    secbert_v2_adverse_evidence_score = VALUES(secbert_v2_adverse_evidence_score),
                    secbert_v2_rf_norm_diff = VALUES(secbert_v2_rf_norm_diff),
                    secbert_v2_mda_cosine_sim = VALUES(secbert_v2_mda_cosine_sim),
                    secbert_v2_mda_semantic_drift = VALUES(secbert_v2_mda_semantic_drift),
                    secbert_v2_rf_mda_divergence = VALUES(secbert_v2_rf_mda_divergence);
                """
                cursor.execute(insert_sql, (
                    ticker, cik, fy, form, f_date, r_date,
                    acc, prior_acc, prior_fdate,
                    status, pretraining_overlap,
                    rf_available, mda_available,
                    has_prior, rf_chunks, rf_tokens,
                    mda_chunks, rf_cosine,
                    rf_drift, novelty_score,
                    adverse_evidence, rf_norm_diff,
                    mda_cosine, mda_drift,
                    rf_mda_div, MODEL_CHECKPOINT
                ))
                mysql_conn.commit()
                total_saved += 1
                already_processed.add(acc)

                if rf_emb is not None:
                    prior_rf_emb = rf_emb
                    prior_rf_chunks = rf_chunk_list
                if mda_emb is not None:
                    prior_mda_emb = mda_emb
                prior_meta = {"accession_number": acc, "filing_date": f_date}

        cursor.close()
        mysql_conn.close()
        elapsed = time.time() - t_start
        logger.info(f"Completed SEC-BERT v2 extraction: {total_saved} filings in {elapsed:.2f}s")


if __name__ == "__main__":
    extractor = SecBertFeatureExtractorV2()
    extractor.extract_all_filings()
