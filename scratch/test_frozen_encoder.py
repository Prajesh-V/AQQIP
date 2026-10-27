import os
import sys
import time
sys.path.insert(0, r"c:\dev\AQQIP")
from pymongo import MongoClient
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from config.settings import get_settings
from src.qualitative.html_cleaner import clean_html, preprocess_text
from src.qualitative.section_extractor import extract_risk_factors, extract_mda

def test_frozen_encoder():
    device = "cpu"
    model_name = "nlpaueb/sec-bert-base"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    settings = get_settings()
    mongo = MongoClient(settings.mongo.uri)
    db = mongo[settings.mongo.database]
    coll = db["sec_filings"]
    
    # Get AAPL 2023 and 2024
    doc_2023 = coll.find_one({"ticker": "AAPL", "fiscal_year": 2023})
    doc_2024 = coll.find_one({"ticker": "AAPL", "fiscal_year": 2024})
    
    def get_section_embedding(doc, section_fn):
        cleaned = preprocess_text(clean_html(doc["raw_content"]))
        res = section_fn(doc["ticker"], doc["accession_number"], cleaned)
        text = res.raw_text
        if not text:
            return None, []
        
        # Tokenize with max_length=512 chunks
        tokens = tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokens["input_ids"][0]
        
        chunk_size = 512
        chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]
        chunk_embeddings = []
        
        with torch.no_grad():
            for c in chunks[:10]: # Take up to first 10 chunks (~5000 tokens covers primary risk disclosures)
                # Pad to 512 if needed
                padded_ids = torch.full((1, chunk_size), tokenizer.pad_token_id, dtype=torch.long, device=device)
                padded_ids[0, :len(c)] = c.to(device)
                attn_mask = (padded_ids != tokenizer.pad_token_id).long()
                
                out = model(input_ids=padded_ids, attention_mask=attn_mask)
                last_hidden = out.last_hidden_state # [1, 512, 768]
                
                # Mean pool over unmasked tokens
                mask_expanded = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                chunk_emb = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
                chunk_embeddings.append(chunk_emb)
                
        if not chunk_embeddings:
            return None, []
        
        section_emb = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(section_emb)
        if norm > 0:
            section_emb = section_emb / norm
        return section_emb, chunk_embeddings

    print("Computing embeddings for AAPL 2023...")
    t0 = time.time()
    e_rf_23, chunks_23 = get_section_embedding(doc_2023, extract_risk_factors)
    t1 = time.time()
    print(f"2023 Item 1A processed in {t1-t0:.2f}s, Chunks: {len(chunks_23)}")
    
    print("Computing embeddings for AAPL 2024...")
    t2 = time.time()
    e_rf_24, chunks_24 = get_section_embedding(doc_2024, extract_risk_factors)
    t3 = time.time()
    print(f"2024 Item 1A processed in {t3-t2:.2f}s, Chunks: {len(chunks_24)}")
    
    # Compute similarity & drift
    cos_sim = float(np.dot(e_rf_23, e_rf_24))
    semantic_drift = 1.0 - cos_sim
    norm_diff = float(np.linalg.norm(e_rf_24 - e_rf_23))
    print(f"Cosine Similarity (2023 vs 2024): {cos_sim:.4f}")
    print(f"Semantic Drift (1 - cos_sim): {semantic_drift:.4f}")
    print(f"Norm Diff: {norm_diff:.4f}")

if __name__ == "__main__":
    test_frozen_encoder()
