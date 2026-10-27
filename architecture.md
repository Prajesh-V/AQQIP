# System Architecture
## Autonomous Quantitative & Qualitative Investment Platform (AQQIP)

**Stack constraint:** $0 cost. All infrastructure runs in local Docker containers or as embedded/local libraries. The only external network calls are to yfinance, SEC EDGAR REST API, and the Google Gemini Free Tier API.

---

## 1. High-Level Container Topology

```
                                   ┌────────────────────────────────────────┐
                                   │              docker-compose             │
                                   │                (aqqip-net)              │
                                   └────────────────────────────────────────┘

  ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────────┐
  │   mysql (3306)   │      │  mongodb (27017)  │      │  chromadb (embedded  │
  │  prices +        │      │  raw SEC filings  │      │  volume-mounted,     │
  │  predictions     │      │  + metadata       │      │  no separate         │
  │                  │      │                    │      │  container required) │
  └────────▲─────────┘      └────────▲──────────┘      └──────────▲───────────┘
           │                          │                            │
           │                          │                            │
  ┌────────┴──────────────────────────┴────────────────────────────┴─────────┐
  │                          ingestion_service (Python)                       │
  │  - yfinance_fetcher.py    → writes OHLCV to MySQL                         │
  │  - edgar_fetcher.py       → writes raw filings to MongoDB                 │
  │  - scheduler (cron / APScheduler)                                         │
  └────────────────────────────────────────────────────────────────────────────┘
           │                                                        │
           ▼                                                        ▼
  ┌──────────────────────────┐                        ┌──────────────────────────┐
  │   quant_engine (Python)   │                        │  qualitative_engine (Py)  │
  │  - Beneish M-Score        │                        │  - NLTK preprocessing     │
  │  - Altman Z-Score         │                        │  - Embeds via HF          │
  │  - LightGBM + Optuna      │                        │    sentence-transformers  │
  │  - logs to MLflow         │                        │  - stores in ChromaDB     │
  └────────────▲──────────────┘                        │  - LangChain agent calls  │
               │                                        │    Gemini API for         │
               │                                        │    semantic-shift summary │
               │                                        └────────────▲───────────────┘
               │                                                     │
               │                                                     │
  ┌────────────┴─────────────────────────────────────────────────────┴────────────┐
  │                            marl_engine (Python)                                │
  │  - Custom Gym environment (Short-Seller / Retail Herd / Market Maker agents)   │
  │  - Consumes quant anomaly score + qualitative shift score as scenario inputs   │
  │  - Outputs simulated max drawdown + expected short profit                      │
  └───────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                          mlops_loop (Python, daily cron)                       │
  │  - Reads open predictions from MySQL                                           │
  │  - Pulls current price via yfinance to resolve target_date predictions         │
  │  - Writes ground truth back to MySQL                                           │
  │  - Triggers LightGBM retrain job + adjusts MARL reward penalty coefficients    │
  └───────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                       streamlit_dashboard (8501)                               │
  │  - Live Accuracy Tracker (reads MySQL predictions table)                       │
  │  - Lie Spotter (reads ChromaDB + Mongo, renders diff)                          │
  │  - MARL Crash Simulation (animated, reads marl_engine output)                  │
  └───────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                       mlflow_ui (5000) — SQLite backend (./mlruns/)            │
  └───────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Docker Compose Services

| Service | Image / Base | Port (host:container) | Volume |
|---|---|---|---|
| `mysql` | `mysql:8.0` | `3306:3306` | `mysql_data:/var/lib/mysql` |
| `mongodb` | `mongo:latest` | `27017:27017` | `mongo_data:/data/db` |
| `ingestion_service` | `python:3.11-slim` | — | `./chroma_data` (shared), source-mounted |
| `quant_engine` | `python:3.11-slim` | — | `./mlruns` |
| `qualitative_engine` | `python:3.11-slim` | — | `./chroma_data` |
| `marl_engine` | `python:3.11-slim` | — | — |
| `mlops_loop` | `python:3.11-slim` (cron-triggered) | — | — |
| `streamlit_dashboard` | `python:3.11-slim` | `8501:8501` | — |
| `mlflow_ui` | `python:3.11-slim` (`mlflow ui`) | `5000:5000` | `./mlruns:/mlruns` |

> ChromaDB runs in **embedded/local persistence mode** (no separate server container) — it is initialized as a Python client with `persist_directory` pointed at a shared Docker volume (`chroma_data`), mounted into both `ingestion_service`/`qualitative_engine` (write) and `streamlit_dashboard` (read).

---

## 2. Database Schemas

### 2.1 MySQL 8.0 — Relational Store (time-series + MLOps)

**Database:** `aqqip`

#### Table: `ohlcv_prices`
| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT AUTO_INCREMENT PK | |
| `ticker` | VARCHAR(16) | indexed |
| `trade_date` | DATE | indexed |
| `open` | DECIMAL(18,4) | |
| `high` | DECIMAL(18,4) | |
| `low` | DECIMAL(18,4) | |
| `close` | DECIMAL(18,4) | |
| `adj_close` | DECIMAL(18,4) | |
| `volume` | BIGINT | |
| `source` | VARCHAR(32) | e.g. `yfinance` |
| `created_at` | TIMESTAMP DEFAULT CURRENT_TIMESTAMP | |
| **UNIQUE** | `(ticker, trade_date)` | prevents duplicate ingestion |

#### Table: `fundamental_ratios`
| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT AUTO_INCREMENT PK | |
| `ticker` | VARCHAR(16) | indexed |
| `fiscal_period` | VARCHAR(8) | e.g. `FY2024`, `Q3-2024` |
| `beneish_m_score` | DECIMAL(10,4) | |
| `altman_z_score` | DECIMAL(10,4) | |
| `raw_features_json` | JSON | intermediate ratio components (DSRI, GMI, AQI, etc.) |
| `computed_at` | TIMESTAMP DEFAULT CURRENT_TIMESTAMP | |
| **UNIQUE** | `(ticker, fiscal_period)` | |

#### Table: `predictions`
| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT AUTO_INCREMENT PK | |
| `ticker` | VARCHAR(16) | indexed |
| `model_run_id` | VARCHAR(64) | FK-equivalent to MLflow `run_id` |
| `predicted_drop_probability` | DECIMAL(6,5) | 0–1 |
| `predicted_at` | TIMESTAMP | when prediction was made |
| `target_date` | DATE | date the prediction is evaluated against |
| `actual_drop_observed` | BOOLEAN NULL | NULL until resolved by MLOps loop |
| `actual_price_change_pct` | DECIMAL(8,4) NULL | |
| `resolved_at` | TIMESTAMP NULL | |
| `status` | ENUM('open','resolved') | default `open` |

#### Table: `marl_simulation_runs`
| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT AUTO_INCREMENT PK | |
| `ticker` | VARCHAR(16) | |
| `scenario_params_json` | JSON | agent counts, liquidity params, reward weights |
| `max_drawdown_pct` | DECIMAL(8,4) | |
| `expected_short_profit_pct` | DECIMAL(8,4) | |
| `run_at` | TIMESTAMP DEFAULT CURRENT_TIMESTAMP | |

---

### 2.2 MongoDB — Unstructured Store

**Database:** `aqqip_raw`

#### Collection: `sec_filings`
```json
{
  "_id": "ObjectId",
  "ticker": "string",
  "cik": "string",
  "filing_type": "10-K | 10-Q",
  "fiscal_year": "number",
  "filing_date": "ISODate",
  "accession_number": "string (unique index)",
  "source_url": "string",
  "raw_text": "string (full filing text, or XBRL-derived text)",
  "sections": {
    "risk_factors": "string",
    "mdna": "string",
    "financial_statements_notes": "string"
  },
  "ingested_at": "ISODate",
  "processing_status": "raw | preprocessed | embedded"
}
```
Index: unique on `accession_number`; compound index on `(ticker, fiscal_year, filing_type)` for chronological retrieval.

#### Collection: `semantic_diffs`
```json
{
  "_id": "ObjectId",
  "ticker": "string",
  "section": "risk_factors",
  "year_from": "number",
  "year_to": "number",
  "cosine_similarity": "number",
  "deleted_sentences": ["string", "..."],
  "added_sentences": ["string", "..."],
  "llm_summary": "string (Gemini-generated interpretation)",
  "computed_at": "ISODate"
}
```

---

### 2.3 ChromaDB — Vector Store

- **Collection naming convention:** `filings_{ticker}` (one collection per ticker to keep chronological filtering cheap).
- **Metadata per embedded chunk:**
  ```json
  {
    "ticker": "string",
    "filing_type": "10-K",
    "fiscal_year": 2024,
    "section": "risk_factors",
    "chunk_index": 0,
    "accession_number": "string"
  }
  ```
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU inference).
- Chronological separation is achieved purely through **metadata filtering** at query time (`where={"fiscal_year": 2023}` vs `{"fiscal_year": 2024}`) rather than separate physical stores — this keeps the RAG flow simple and avoids duplicate embedding pipelines.

---

## 3. RAG Flow — Chronological Semantic-Shift Detection

```
1. New filing ingested (MongoDB, status=raw)
        │
        ▼
2. NLTK preprocessing
   - sentence tokenization
   - extraction of "Risk Factors" section via section-heading regex
        │
        ▼
3. Chunking (section → paragraph-level chunks, ~200-400 tokens)
        │
        ▼
4. Embedding (all-MiniLM-L6-v2, local CPU inference)
        │
        ▼
5. Upsert into ChromaDB collection `filings_{ticker}`
   metadata tagged with fiscal_year (chronological tag)
        │
        ▼
6. On-demand comparison (triggered by quant engine flag OR analyst request):
   a. Retrieve all Year N chunks (metadata filter fiscal_year=N)
   b. Retrieve all Year N-1 chunks (metadata filter fiscal_year=N-1)
   c. Compute pairwise cosine similarity between nearest-neighbor chunk pairs
   d. Flag chunks below similarity threshold (default 0.75) as "shifted"
   e. Identify chunks present in Year N-1 with no close match in Year N → "deleted disclosure"
   f. Identify chunks present in Year N with no close match in Year N-1 → "new disclosure"
        │
        ▼
7. LangChain Agent (backed by Gemini API) receives the deleted/added/shifted
   sentence sets and produces a natural-language interpretation:
   "Company removed language regarding [X] which was present in the prior
   filing, and added hedged language regarding [Y]."
        │
        ▼
8. Result persisted to MongoDB (`semantic_diffs` collection)
        │
        ▼
9. Streamlit "Lie Spotter" reads `semantic_diffs` + raw chunk text to render
   a highlighted diff view (deletions in red, additions in green).
```

---

## 4. Data Flow Summary (End-to-End)

```
[yfinance] ──▶ [ingestion_service] ──▶ [MySQL: ohlcv_prices]
                                                │
[SEC EDGAR] ─▶ [ingestion_service] ──▶ [MongoDB: sec_filings]
                                                │
                                                ▼
                              [quant_engine] computes ratios ──▶ [MySQL: fundamental_ratios]
                                                │
                              [qualitative_engine] embeds text ──▶ [ChromaDB]
                                                │
                              [qualitative_engine] diffs years ──▶ [MongoDB: semantic_diffs]
                                                │
                                                ▼
                         [quant_engine] LightGBM (Optuna-tuned) trains on
                         ratios + semantic-shift score as features
                                                │
                                     logs run ──▶ [MLflow (SQLite)]
                                                │
                                    prediction ──▶ [MySQL: predictions]
                                                │
                                                ▼
                              [marl_engine] runs simulation using
                              anomaly score + shift score as scenario seed
                                                │
                                   result ──▶ [MySQL: marl_simulation_runs]
                                                │
                                                ▼
                    [mlops_loop] (daily cron) resolves predictions vs.
                    actual yfinance outcome, feeds hard negatives back
                    into next LightGBM retrain + adjusts MARL reward weights
                                                │
                                                ▼
                         [streamlit_dashboard] reads MySQL + MongoDB +
                         ChromaDB directly (read-only) to render all views
```

---

## 5. Network & Security Notes (Local-First)

- All inter-service communication happens over the internal Docker bridge network (`aqqip-net`); only `streamlit_dashboard` (8501) and `mlflow_ui` (5000) expose ports to the host.
- MySQL and MongoDB credentials are supplied via `.env` (never committed) and consumed via Docker Compose `env_file`.
- The Gemini API key is the **only** secret requiring external network egress; it is used exclusively by `qualitative_engine`'s LangChain agent step.
- SEC EDGAR requires a compliant `User-Agent` header (e.g., `AQQIP research-bot contact@example.com`) — no API key needed, but this header is mandatory to avoid being blocked.
