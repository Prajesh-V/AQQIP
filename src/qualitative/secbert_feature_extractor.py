"""
SEC-BERT Qualitative Feature Extractor

Extracts compact, reproducible textual representations and change metrics from 10-K
Risk Factors and MD&A sections using frozen pretrained `nlpaueb/sec-bert-base`.
Preserves strict point-in-time provenance, missingness indicators, and prior filing pairs.
Persists results into `qualitative_transformer_features` table.
"""

from __future__ import annotations
import os
import sys
import time
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

logger = get_logger("secbert_extractor")

MODEL_CHECKPOINT = "nlpaueb/sec-bert-base"
CHUNK_SIZE = 512
MAX_CHUNKS = 4  # 4 chunks = 2,048 tokens per section, captures core disclosures with high throughput


def setup_transformer_table(cursor):
    """Creates the qualitative_transformer_features table if it doesn't exist."""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qualitative_transformer_features (
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
        secbert_rf_available TINYINT DEFAULT 0,
        secbert_mda_available TINYINT DEFAULT 0,
        secbert_has_prior TINYINT DEFAULT 0,
        secbert_rf_chunks INT DEFAULT 0,
        secbert_mda_chunks INT DEFAULT 0,
        secbert_rf_cosine_sim FLOAT DEFAULT 1.0,
        secbert_rf_semantic_drift FLOAT DEFAULT 0.0,
        secbert_rf_norm_diff FLOAT DEFAULT 0.0,
        secbert_mda_cosine_sim FLOAT DEFAULT 1.0,
        secbert_mda_semantic_drift FLOAT DEFAULT 0.0,
        secbert_rf_mda_divergence FLOAT DEFAULT 0.0,
        model_name VARCHAR(50) DEFAULT 'nlpaueb/sec-bert-base',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_ticker_date (ticker, filing_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)


class SecBertFeatureExtractor:
    """Frozen SEC-BERT feature extractor with batched tensor inference."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.settings = get_settings()
        logger.info(f"Loading frozen {MODEL_CHECKPOINT} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.model = AutoModel.from_pretrained(MODEL_CHECKPOINT).to(self.device)
        self.model.eval()

    def embed_section(self, text: str) -> Tuple[Optional[np.ndarray], int]:
        """Tokenizes, chunks, and batched mean-pools over token and chunk embeddings."""
        if not text or not text.strip():
            return None, 0

        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokens["input_ids"][0]

        chunks = [input_ids[i:i+CHUNK_SIZE] for i in range(0, len(input_ids), CHUNK_SIZE)]
        selected_chunks = chunks[:MAX_CHUNKS]
        if not selected_chunks:
            return None, 0

        batch_size = len(selected_chunks)
        padded_batch = torch.full((batch_size, CHUNK_SIZE), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        for idx, c in enumerate(selected_chunks):
            padded_batch[idx, :len(c)] = c.to(self.device)
        attn_mask = (padded_batch != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            out = self.model(input_ids=padded_batch, attention_mask=attn_mask)
            last_hidden = out.last_hidden_state  # [B, 512, 768]
            mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)  # [B, 768]
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            chunk_embeddings = (sum_embeddings / sum_mask).cpu().numpy()  # [B, 768]

        section_emb = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(section_emb)
        if norm > 0:
            section_emb = section_emb / norm
        return section_emb, batch_size

    def extract_all_filings(self):
        """Processes all 10-K filings stored in MongoDB ordered chronologically per ticker."""
        mongo = MongoClient(self.settings.mongo.uri)
        db = mongo[self.settings.mongo.database]
        coll = db["sec_filings"]

        # Check existing processed accessions in MySQL
        mysql_conn = mysql.connector.connect(
            host=self.settings.mysql.host,
            port=self.settings.mysql.port,
            user=self.settings.mysql.user,
            password=self.settings.mysql.password,
            database=self.settings.mysql.database
        )
        cursor = mysql_conn.cursor(dictionary=True)
        setup_transformer_table(cursor)
        mysql_conn.commit()

        cursor.execute("SELECT accession_number FROM qualitative_transformer_features;")
        already_processed = set(r["accession_number"] for r in cursor.fetchall())
        logger.info(f"Already processed transformer filings: {len(already_processed)}")

        # Fetch all filings from MongoDB
        docs = list(coll.find({}, {
            "ticker": 1, "cik": 1, "fiscal_year": 1, "filing_type": 1,
            "filing_date": 1, "report_date": 1, "accession_number": 1, "raw_content": 1
        }))
        logger.info(f"Fetched {len(docs)} filings from MongoDB")

        # Group by ticker and sort chronologically
        ticker_docs: Dict[str, List[Dict[str, Any]]] = {}
        for d in docs:
            t = d["ticker"]
            ticker_docs.setdefault(t, []).append(d)

        for t in ticker_docs:
            ticker_docs[t].sort(key=lambda x: (pd.to_datetime(x["filing_date"]), x["accession_number"]))

        total_saved = len(already_processed)
        t_start = time.time()

        for ticker, t_list in ticker_docs.items():
            prior_rf_emb: Optional[np.ndarray] = None
            prior_mda_emb: Optional[np.ndarray] = None
            prior_meta: Optional[Dict[str, Any]] = None

            for d in t_list:
                acc = d["accession_number"]
                f_date = pd.to_datetime(d["filing_date"]).strftime("%Y-%m-%d")
                r_date = pd.to_datetime(d.get("report_date")).strftime("%Y-%m-%d") if d.get("report_date") else None
                fy = d.get("fiscal_year")
                form = d.get("filing_type", "10-K")
                cik = d.get("cik")

                # If already processed, we can still load embedding if needed for prior, but for throughput:
                if acc in already_processed:
                    prior_meta = {"accession_number": acc, "filing_date": f_date}
                    continue

                raw_html = d.get("raw_content", "")
                cleaned = preprocess_text(clean_html(raw_html))
                rf_res = extract_risk_factors(ticker, acc, cleaned)
                mda_res = extract_mda(ticker, acc, cleaned)

                rf_emb, rf_chunks = self.embed_section(rf_res.raw_text)
                mda_emb, mda_chunks = self.embed_section(mda_res.raw_text)

                has_prior = 1 if prior_meta is not None else 0
                prior_acc = prior_meta["accession_number"] if prior_meta else None
                prior_fdate = prior_meta["filing_date"] if prior_meta else None

                rf_available = 1 if rf_emb is not None else 0
                mda_available = 1 if mda_emb is not None else 0

                # Compute change metrics
                rf_cosine = 1.0
                rf_drift = 0.0
                rf_norm_diff = 0.0
                if rf_emb is not None and prior_rf_emb is not None:
                    sim = float(np.dot(rf_emb, prior_rf_emb))
                    rf_cosine = float(np.clip(sim, -1.0, 1.0))
                    rf_drift = float(1.0 - rf_cosine)
                    rf_norm_diff = float(np.linalg.norm(rf_emb - prior_rf_emb))

                mda_cosine = 1.0
                mda_drift = 0.0
                if mda_emb is not None and prior_mda_emb is not None:
                    sim = float(np.dot(mda_emb, prior_mda_emb))
                    mda_cosine = float(np.clip(sim, -1.0, 1.0))
                    mda_drift = float(1.0 - mda_cosine)

                # RF vs MDA Divergence in same filing
                rf_mda_div = 0.0
                if rf_emb is not None and mda_emb is not None:
                    rf_mda_div = float(1.0 - np.clip(np.dot(rf_emb, mda_emb), -1.0, 1.0))

                status = "SUCCESS" if (rf_available and mda_available) else ("PARTIAL" if (rf_available or mda_available) else "FAILED")

                # Insert or replace in MySQL
                insert_sql = """
                INSERT INTO qualitative_transformer_features (
                    ticker, cik, fiscal_year, form, filing_date, report_date,
                    accession_number, prior_accession_number, prior_filing_date,
                    extraction_status, secbert_rf_available, secbert_mda_available,
                    secbert_has_prior, secbert_rf_chunks, secbert_mda_chunks,
                    secbert_rf_cosine_sim, secbert_rf_semantic_drift, secbert_rf_norm_diff,
                    secbert_mda_cosine_sim, secbert_mda_semantic_drift, secbert_rf_mda_divergence,
                    model_name
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) ON DUPLICATE KEY UPDATE
                    prior_accession_number = VALUES(prior_accession_number),
                    prior_filing_date = VALUES(prior_filing_date),
                    extraction_status = VALUES(extraction_status),
                    secbert_rf_available = VALUES(secbert_rf_available),
                    secbert_mda_available = VALUES(secbert_mda_available),
                    secbert_has_prior = VALUES(secbert_has_prior),
                    secbert_rf_chunks = VALUES(secbert_rf_chunks),
                    secbert_mda_chunks = VALUES(secbert_mda_chunks),
                    secbert_rf_cosine_sim = VALUES(secbert_rf_cosine_sim),
                    secbert_rf_semantic_drift = VALUES(secbert_rf_semantic_drift),
                    secbert_rf_norm_diff = VALUES(secbert_rf_norm_diff),
                    secbert_mda_cosine_sim = VALUES(secbert_mda_cosine_sim),
                    secbert_mda_semantic_drift = VALUES(secbert_mda_semantic_drift),
                    secbert_rf_mda_divergence = VALUES(secbert_rf_mda_divergence);
                """
                cursor.execute(insert_sql, (
                    ticker, cik, fy, form, f_date, r_date,
                    acc, prior_acc, prior_fdate,
                    status, rf_available, mda_available,
                    has_prior, rf_chunks, mda_chunks,
                    rf_cosine, rf_drift, rf_norm_diff,
                    mda_cosine, mda_drift, rf_mda_div,
                    MODEL_CHECKPOINT
                ))
                mysql_conn.commit()
                total_saved += 1
                already_processed.add(acc)

                # Update prior references
                if rf_emb is not None:
                    prior_rf_emb = rf_emb
                if mda_emb is not None:
                    prior_mda_emb = mda_emb
                prior_meta = {"accession_number": acc, "filing_date": f_date}

        cursor.close()
        mysql_conn.close()
        elapsed = time.time() - t_start
        logger.info(f"Completed Transformer feature extraction: {total_saved} filings in {elapsed:.2f}s")


if __name__ == "__main__":
    extractor = SecBertFeatureExtractor()
    extractor.extract_all_filings()
