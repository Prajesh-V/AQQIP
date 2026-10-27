# AQQIP — Complete Project Handover, Repository Audit, Research State & Continuation Context

**Repository Path:** `C:\dev\AQQIP`  
**Governing Document Version:** 1.0 (Comprehensive Handover)  
**Generation Timestamp:** 2026-10-27T17:00:00Z (UTC)  
**Auditor / Agent:** Antigravity System Agent (Advanced Agentic Pair-Programmer)  
**Safety Classification:** Read-Only Audit & Documentation Archive  

---

## Table of Contents
1. [Repository Identity, Purpose, and Research Objectives](#1-repository-identity-purpose-and-research-objectives)
2. [End-to-End Operational Pipeline](#2-end-to-end-operational-pipeline)
3. [Architecture and Multi-Store Persistence Infrastructure](#3-architecture-and-multi-store-persistence-infrastructure)
4. [Complete Codebase & File Inventory](#4-complete-codebase--file-inventory)
5. [Quantitative and Machine Learning Pipeline Audit](#5-quantitative-and-machine-learning-pipeline-audit)
6. [Qualitative NLP, Embeddings, and Evidence Pipeline](#6-qualitative-nlp-embeddings-and-evidence-pipeline)
7. [Multi-Agent Reinforcement Learning (MARL) Market Simulation](#7-multi-agent-reinforcement-learning-marl-market-simulation)
8. [MLOps Feedback Loop & Prediction Resolution](#8-mlops-feedback-loop--prediction-resolution)
9. [Authoritative Research Metrics & Governance Closure](#9-authoritative-research-metrics--governance-closure)
10. [Frontend Architecture, Pending Integration & Visual Constraints](#10-frontend-architecture-pending-integration--visual-constraints)
11. [Known Engineering Milestones & Historical Context](#11-known-engineering-milestones--historical-context)
12. [Actionable Status Register and Prioritized Roadmap](#12-actionable-status-register-and-prioritized-roadmap)
13. [Unresolved Questions, Contradictions, and Governance Risks](#13-unresolved-questions-contradictions-and-governance-risks)
14. [COPY THIS INTO THE NEXT CHATGPT CONVERSATION](#14-copy-this-into-the-next-chatgpt-conversation)

---

## 1. Repository Identity, Purpose, and Research Objectives

### 1.1 Formal Identity & Mission
* **Full Name:** Autonomous Quantitative & Qualitative Investment Platform (AQQIP).
* **Working Research Subtitle:** *A Multi-Agent LLM Framework for Autonomous Financial Forensics and Market Simulation*.
* **Core Product Metaphor:** **"Analyst Force Multiplier."** A local-first research platform designed to automate the triage of public corporate filings, quantify accounting red flags, detect semantic disclosure softening or omission, simulate market microstructure crash dynamics under adversarial short selling, and autonomously grade its predictions against real market outcomes.
* **Fundamental Cost Constraint:** **$0.00 Total Spend.** The system is strictly engineered to run locally via Docker Compose, requiring zero commercial cloud infrastructure (no AWS/GCP/Azure compute or hosted databases) and zero paid API services. The only permitted external LLM call is the Google Gemini Free Tier API (via Google AI Studio). All embeddings run CPU-locally, and local LLM workflows target Ollama (`qwen3:8b` primary, `phi4-mini` fallback).

### 1.2 Target Personas & Use Cases
1. **Forensic Research Analyst:** Screening large equity universes down to anomalous corporate issuers exhibiting earnings manipulation or liquidity strain.
2. **Adversarial Short-Seller / Research Boutique:** Evaluating the market depth and plausible short profit/drawdown distribution of a thesis before capital commitment.
3. **Internal Risk & Compliance Auditor:** Monitoring subsidiaries or counterparty holdings for disclosure drift and statistical accounting anomalies.
4. **Quantitative / MLOps Researcher:** Investigating leak-free multi-modal evidence fusion, ranking loss functions, and self-correcting feedback loops.

### 1.3 Boundary Clarification: Prototype vs. Research vs. Non-Goals
* **Not an Investment Advisor or Live Execution Brokerage:** AQQIP produces research signals, forensic scores, and simulated microstructure trajectories. It **never** connects to brokerage APIs, executes real-money trades, or produces prescriptive "buy/sell" investment advice. All outputs require mandatory disclaimers.
* **Universe & Coverage Limits:** The audited universe focuses on a balanced cross-section of $N = 32$ liquid US public equities with SEC EDGAR filings (Forms 10-K and 10-Q) and daily OHLCV pricing from `yfinance`. International jurisdictions and non-standard filings (8-K, S-1, proxy statements) are explicitly out of scope for v1.

---

## 2. End-to-End Operational Pipeline

The platform is designed around a continuous, self-correcting 6-stage lifecycle:

$$\text{DATA} \longrightarrow \text{EVIDENCE} \longrightarrow \text{SIGNALS} \longrightarrow \text{SCENARIO} \longrightarrow \text{OUTCOME} \longrightarrow \text{FEEDBACK}$$

```
   ┌──────────────────────────────────────────────────────────────┐
   │                           1. DATA                            │
   │  Daily OHLCV (yfinance) → MySQL [ohlcv_prices]               │
   │  SEC EDGAR Filings (Submissions & Facts) → MongoDB           │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                         2. EVIDENCE                          │
   │  Quant: Beneish M-Score & Altman Z-Score → [fundamental_ratios│
   │  Qual: HTML cleaning, Item 1A/7 parsing, SEC-BERT embeddings │
   │  Qual: Directional adversity vs. benign expansion lexicons   │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                          3. SIGNALS                          │
   │  Point-in-Time Joiner (acceptance date integrity)            │
   │  LightGBM Anomaly Classifier (Optuna tuned, MLflow logged)  │
   │  Canonical SignalBundle Construction (cryptographic hash)    │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                         4. SCENARIO                          │
   │  MARL Simulation Engine (Custom Microstructure Gym Env)      │
   │  Short-Seller vs. Retail Herd vs. Inventory-Defending MM     │
   │  Outputs: Max Drawdown %, Expected Short Profit %            │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                         5. OUTCOME                           │
   │  Inference logging to MySQL [predictions] (target_date = t+20)│
   │  Streamlit Dashboard Visualization (:8501)                   │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                         6. FEEDBACK                          │
   │  MLOpsResolver queries MySQL prices at target_date maturity  │
   │  Calculates actual drop & forward return %                   │
   │  Updates prediction status ('open' → 'resolved')             │
   │  Triggers retraining loop & adjusts MARL policy penalties    │
   └──────────────────────────────────────────────────────────────┘
```

* **Implemented & Verified:** Stages 1, 2, 3, 4, and 5 (inference & resolution backend).
* **Partially Implemented:** Stage 5 (Streamlit dashboard has container healthcheck stub; UI business views pending).
* **Active Research:** Stage 3 ranking objective evolution (Phase 3 LambdaMART NDCG@2 vs. Point-wise Log-Loss).

---

## 3. Architecture and Multi-Store Persistence Infrastructure

### 3.1 Container Topology & Host Bindings
The system runs via [`docker-compose.yml`](file:///c:/dev/AQQIP/docker-compose.yml) over the isolated bridge network `aqqip-net`:

| Service Name | Container Name | Base Image / Build | Port Mapping | Volume / Persistence | Healthcheck |
|---|---|---|---|---|---|
| `mysql` | `aqqip-mysql` | `mysql:8.0` | `3306:3306` | `mysql_data:/var/lib/mysql`, `db/mysql/init:/docker-entrypoint-initdb.d:ro` | `mysqladmin ping` (10s interval) |
| `mongodb` | `aqqip-mongodb` | `mongo:7.0` | `27017:27017` | `mongo_data:/data/db`, `db/mongo/init/init-indexes.js:/...:ro` | `mongosh adminCommand('ping')` |
| `mlflow_ui` | `aqqip-mlflow` | `Dockerfile.python` | `5000:5000` | `mlflow_data:/mlruns` | `curl -f http://localhost:5000/health` |
| `ingestion_service`| `aqqip-ingestion`| `Dockerfile.python` | None (Worker) | `chroma_data:/app/chroma_data`, bind mounts `./src`, `./config` | Stub process supervisor |
| `quant_engine` | `aqqip-quant` | `Dockerfile.python` | None (Worker) | `mlflow_data:/mlruns`, `./src`, `./config` | Stub process supervisor |
| `qualitative_engine`| `aqqip-qualitative`| `Dockerfile.python` | None (Worker) | `chroma_data:/app/chroma_data`, `./src`, `./config` | Stub process supervisor |
| `marl_engine` | `aqqip-marl` | `Dockerfile.python` | None (Worker) | Bind mounts `./src`, `./config` | Stub process supervisor |
| `mlops_loop` | `aqqip-mlops` | `Dockerfile.python` | None (Cron) | Bind mounts `./src`, `./config` | Stub process supervisor |
| `streamlit_dashboard`| `aqqip-dashboard`| `Dockerfile.python` | `8501:8501` | Bind mounts `./src`, `./config`, `./dashboard` | `curl -f http://localhost:8501/_stcore/health` |

### 3.2 Database Schemas & Storage Details
1. **MySQL 8.0 (`aqqip` database):**
   * [`ohlcv_prices`](file:///c:/dev/AQQIP/db/mysql/init/001_create_ohlcv_prices.sql): `id`, `ticker`, `trade_date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `source`, `created_at`. Constraint: `UNIQUE KEY uq_ticker_trade_date (ticker, trade_date)`.
   * [`fundamental_ratios`](file:///c:/dev/AQQIP/db/mysql/init/002_create_fundamental_ratios.sql): `id`, `ticker`, `fiscal_period`, `beneish_m_score`, `altman_z_score`, `raw_features_json`, `computed_at`. Constraint: `UNIQUE KEY uq_ticker_fiscal_period (ticker, fiscal_period)`.
   * [`predictions`](file:///c:/dev/AQQIP/db/mysql/init/003_create_predictions.sql): `id`, `ticker`, `model_run_id`, `predicted_drop_probability`, `predicted_at`, `target_date`, `actual_drop_observed`, `actual_price_change_pct`, `resolved_at`, `status` (`ENUM('open','resolved','invalid_target')`), `input_feature_hash`. Indexes on `ticker`, `target_date`, `status`.
   * [`marl_simulation_runs`](file:///c:/dev/AQQIP/db/mysql/init/004_create_marl_simulation_runs.sql): `id`, `simulation_id` (UUID), `ticker`, `scenario_params_json`, `max_drawdown_pct`, `expected_short_profit_pct`, `run_at`.
2. **MongoDB 7.0 (`aqqip_raw` database):**
   * `sec_filings`: Stores raw HTML and cleaned text of 10-K/10-Q filings. Indexed on `accession_number` (unique) and compound index `(ticker, filing_date)`.
   * `semantic_diffs`: Stores chunk-level additions, deletions, cosine similarity matrices, and LLM-generated shift narratives. Unique index on `(ticker, prior_accession, current_accession)`.
3. **ChromaDB (Embedded Local Persistence):**
   * Configured path: `./chroma_data`. Embedded mode (no separate network daemon).
   * Schema: Partitioned per-ticker collections (`filings_{ticker}`) to enforce strict isolation and avoid cross-firm semantic leakage.
4. **MLflow (Local SQLite Backend):**
   * Tracking URI: `sqlite:///mlruns/mlflow.db`. Artifact root: `./mlruns/artifacts`.
   * Tracks hyperparameter trials (Optuna studies), feature ordering, LightGBM binary booster files, and validation metrics.

---

## 4. Complete Codebase & File Inventory

### 4.1 Git and Working Tree Status
* **Git Version Control Status:** The directory `C:\dev\AQQIP` is **not currently initialized as a Git repository** (no `.git` directory exists on disk or in parent folders).
* **Working Tree State:** All files are untracked local filesystem assets. All files, historical artifacts, and research manifests must be preserved without in-place destruction.

### 4.2 Module-by-Module Directory Inventory

```
C:\dev\AQQIP\
├── config/                         Centralized configuration
│   ├── settings.py                 Typed Pydantic settings loading from .env
│   └── __init__.py
├── dashboard/                      Visualization layer
│   └── app.py                      Streamlit entrypoint (Phase 1 container stub)
├── db/                             Database initialization scripts
│   ├── mongo/init/init-indexes.js  MongoDB index setup (accession, ticker_year)
│   └── mysql/init/                 Numbered SQL migrations (001 to 004)
├── docs/                           Documentation & Handover
│   └── handover/                   Primary handover artifacts (Deliverables A, B, C, D)
├── eval_artifacts/                 Frozen evaluation artifacts
│   ├── authoritative_eval_predictions_2025_2026.parquet  (Diagnostic holdout panel)
│   └── authoritative_eval_predictions_2025_2026.csv      (Diagnostic holdout panel)
├── experiments/                    Empirical research experiment outputs
│   ├── asymmetric_volatility_phase2_20261021_203027/     (Failed Invocation 1 - empty dirs)
│   └── asymmetric_volatility_phase2_20261021_203158/     (Audited Invocation 2 - 26 files)
├── manifests/                      Integrity freeze manifests
│   └── pre_training_freeze_manifest_phase1c.json          (Pre-training checksum freeze)
├── mlruns/                         MLflow tracking SQLite store & artifacts (1,516 files)
├── scratch/                        Exploratory analysis scripts (audit, checks, diagnostics)
├── src/                            Application source modules
│   ├── database/                   Client connection managers (mysql_client, mongo_client)
│   ├── ingestion/                  Market & SEC data collectors, schedulers, validators
│   ├── quant/                      Forensic scorers, dataset builders, LightGBM trainers
│   ├── qualitative/                NLP cleaners, section parsers, SEC-BERT extractors
│   ├── marl/                       Gym environment, policies, engine, signal bridge
│   ├── mlops/                      Prediction resolver, repository, target reconciliation
│   └── utils/                      Structured logging, Tenacity retry decorators, timestamps
└── tests/                          Pytest test suite (38 files, 155 test functions)
```

### 4.3 Detailed Source File Audit

| File Path | Component / Role | Status | Dependencies | Description |
|---|---|---|---|---|
| [`config/settings.py`](file:///c:/dev/AQQIP/config/settings.py) | Configuration | Active | `pydantic`, `pydantic_settings` | Typed configuration loader reading `.env`. Defines connection parameters for MySQL, Mongo, Chroma, MLflow, Ollama, Gemini, and MARL. |
| [`src/database/mysql_client.py`](file:///c:/dev/AQQIP/src/database/mysql_client.py) | Persistence | Active | `mysql.connector`, `config.settings` | Connection context manager providing pooled MySQL connections with dictionary cursors. |
| [`src/database/mongo_client.py`](file:///c:/dev/AQQIP/src/database/mongo_client.py) | Persistence | Active | `pymongo`, `config.settings` | MongoDB client provider for document stores (`sec_filings`, `semantic_diffs`). |
| [`src/ingestion/yfinance_fetcher.py`](file:///c:/dev/AQQIP/src/ingestion/yfinance_fetcher.py) | Ingestion | Active | `yfinance`, `tenacity`, `mysql_client` | Daily OHLCV price fetcher with data validation (range checks, positive volume) and idempotent MySQL upserts. |
| [`src/ingestion/edgar_fetcher.py`](file:///c:/dev/AQQIP/src/ingestion/edgar_fetcher.py) | Ingestion | Active | `requests`, `tenacity`, `mongo_client` | Queries SEC EDGAR submissions API for Forms 10-K/10-Q, validates user-agent compliance, upserts to MongoDB. |
| [`src/ingestion/edgar_company_facts.py`](file:///c:/dev/AQQIP/src/ingestion/edgar_company_facts.py) | Ingestion | Active | `requests`, `tenacity` | Ingests structured XBRL financial facts from SEC EDGAR for fundamental accounting calculation. |
| [`src/ingestion/ticker_cik_resolver.py`](file:///c:/dev/AQQIP/src/ingestion/ticker_cik_resolver.py) | Ingestion | Active | SEC JSON maps | Resolves ticker symbols to 10-digit padded SEC CIK numbers with local cache fallback. |
| [`src/ingestion/edgar_monitor.py`](file:///c:/dev/AQQIP/src/ingestion/edgar_monitor.py) | Ingestion | Active | `edgar_fetcher` | Near-real-time polling daemon monitoring EDGAR RSS/Submissions for new prospective filings. |
| [`src/quant/beneish_score.py`](file:///c:/dev/AQQIP/src/quant/beneish_score.py) | Quant / Forensic | Active | `numpy`, `pandas` | Computes 8-variable Beneish M-Score (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA) to detect earnings manipulation. |
| [`src/quant/altman_score.py`](file:///c:/dev/AQQIP/src/quant/altman_score.py) | Quant / Forensic | Active | `numpy`, `pandas` | Computes Altman Z-Score for manufacturing and non-manufacturing firms to assess solvency distress. |
| [`src/quant/point_in_time_join.py`](file:///c:/dev/AQQIP/src/quant/point_in_time_join.py) | Quant / Data Hygiene| Active | `pandas` | Merges fundamental and qualitative metrics with market prices strictly on filing publication date (zero lookahead). |
| [`src/quant/temporal_purging.py`](file:///c:/dev/AQQIP/src/quant/temporal_purging.py) | Quant / Data Hygiene| Active | `pandas` | Purges observation windows crossing fold boundaries ($H=20$) to eliminate serial correlation leakage. |
| [`src/quant/leakage_validator.py`](file:///c:/dev/AQQIP/src/quant/leakage_validator.py) | Quant / Validation | Active | `pandas` | Programmatic verification ensuring no test fold observation timestamp precedes training fold cutoffs. |
| [`src/quant/lightgbm_trainer.py`](file:///c:/dev/AQQIP/src/quant/lightgbm_trainer.py) | Quant / Modeling | Active | `lightgbm`, `optuna`, `mlflow` | Trains gradient-boosted trees predicting sharp downside events; logs runs, parameters, metrics, and booster artifacts. |
| [`src/quant/asymmetric_volatility_experiment.py`](file:///c:/dev/AQQIP/src/quant/asymmetric_volatility_experiment.py) | Quant / Research | Audited / Frozen | `lightgbm`, `numpy`, `pandas` | Core protocol implementation for Phase 2 specification v4.7: Candidate schemas A, B, C, D; moving block bootstrap; gates G1–G7. |
| [`src/quant/run_asymmetric_volatility_experiment.py`](file:///c:/dev/AQQIP/src/quant/run_asymmetric_volatility_experiment.py) | Quant / Research | Audited / Frozen | `asymmetric_volatility_experiment` | Orchestrator script executing the four-fold rolling-origin evaluation into isolated run directories. |
| [`src/quant/authoritative_eval_artifact.py`](file:///c:/dev/AQQIP/src/quant/authoritative_eval_artifact.py) | Quant / Research | Audited / Active | `mlflow`, `pandas` | Generates authoritative evaluation records for the 2025–2026 diagnostic holdout across M, MF, MFQ, M_Regularized. |
| [`src/quant/inference_pipeline.py`](file:///c:/dev/AQQIP/src/quant/inference_pipeline.py) | Quant / Production | Active | `mlflow`, `mysql_client` | Loads registered model artifact (`runs:/df161...`), builds leak-free features, logs open predictions to MySQL. |
| [`src/qualitative/html_cleaner.py`](file:///c:/dev/AQQIP/src/qualitative/html_cleaner.py) | NLP / Preprocess | Active | `BeautifulSoup4`, `re` | Strips raw HTML and inline XBRL markup from SEC filings prior to regex or token-based section extraction. |
| [`src/qualitative/section_extractor.py`](file:///c:/dev/AQQIP/src/qualitative/section_extractor.py) | NLP / Preprocess | Active | `re` | Extracts Item 1A (Risk Factors) and Item 7 (MD&A) with length fallbacks ($< 500$ words triggers secondary strategy). |
| [`src/qualitative/directional_signal.py`](file:///c:/dev/AQQIP/src/qualitative/directional_signal.py) | NLP / Forensic | Active | `re`, `pydantic` | Disentangles unsigned change magnitude from directional adversity using Loughran-McDonald inspired lexicons. |
| [`src/qualitative/secbert_feature_extractor_v2.py`](file:///c:/dev/AQQIP/src/qualitative/secbert_feature_extractor_v2.py) | NLP / Embeddings | Active | `transformers`, `torch` | Computes SEC-BERT domain embeddings and adverse semantic projection vectors on CPU without lookahead. |
| [`src/qualitative/chroma_store.py`](file:///c:/dev/AQQIP/src/qualitative/chroma_store.py) | NLP / Vector DB | Active | `chromadb` | Embedded vector store manager creating and querying isolated `filings_{ticker}` collections. |
| [`src/marl/engine.py`](file:///c:/dev/AQQIP/src/marl/engine.py) | Simulation / MARL | Active | `numpy`, `gymnasium` | Custom market microstructure simulation engine with 3 agent classes, inventory bounds, and strictly positive prices. |
| [`src/marl/policies.py`](file:///c:/dev/AQQIP/src/marl/policies.py) | Simulation / MARL | Active | `numpy` | Implements Short-Seller (signal-driven), Retail Herd (momentum/panic), and Market Maker (inventory defense) policies. |
| [`src/marl/signal_bridge.py`](file:///c:/dev/AQQIP/src/marl/signal_bridge.py) | Simulation / Bridge | Active | `mlflow`, `mysql_client` | Binds point-in-time market data, ratios, qualitative risk, and MLflow model predictions into canonical `SignalBundle`. |
| [`src/mlops/resolver.py`](file:///c:/dev/AQQIP/src/mlops/resolver.py) | MLOps / Feedback | Active | `mysql_client` | Evaluates open predictions once `target_date <= current_date`, calculating actual return % and updating MySQL rows. |
| [`src/mlops/legacy_reconciliation.py`](file:///c:/dev/AQQIP/src/mlops/legacy_reconciliation.py) | MLOps / Audit | Active | `mysql_client` | Audits historical predictions against the exact 20-trading-day target contract, flagging nonconforming records. |
| [`dashboard/app.py`](file:///c:/dev/AQQIP/dashboard/app.py) | Frontend | Placeholder Stub | `streamlit` | Minimal Phase 1 Streamlit entrypoint; business views (Accuracy Tracker, Lie Spotter, MARL Crash) remain pending. |

---

## 5. Quantitative and Machine Learning Pipeline Audit

### 5.1 Historical Feature Engineering & Target Construction
* **Market Features (9 standard):** `return_1d`, `return_5d`, `return_20d`, `return_60d`, `volatility_20d`, `daily_range_pct`, `avg_range_20d`, `volume_z_score_20d`, `drawdown_60d`.
* **Cross-Sectional Relative Features (M_Regularized):**
  * `return_20d_rel`: Security return minus session cross-sectional mean return.
  * `vol_ratio_20d`: Security volatility divided by session cross-sectional mean volatility.
* **Target Definition & Maturity:**
  * Forward horizon: Exactly $H = 20$ subsequent observed trading sessions in MySQL `ohlcv_prices`.
  * Downside event label: Binary indicator $\mathbb{1}(\text{forward\_return\_pct} \le \tau)$, where $\tau$ corresponds to the unconditional 10th percentile ($\alpha = 0.10$) calibrated exclusively on the training fold.
  * Price adjustment: Enforces `adj_close` to prevent split/dividend false anomalies.

### 5.2 Chronological Validation & Leakage Safeguards
* **Temporal Purging:** When training on pre-fold data, the final 20 trading sessions immediately prior to the test boundary are purged to prevent forward-return label overlap.
* **Point-in-Time Join Integrity:** Fundamentals are merged using the filing's SEC acceptance timestamp, ensuring accounting metrics are inaccessible to the model prior to public disclosure.

---

## 6. Qualitative NLP, Embeddings, and Evidence Pipeline

### 6.1 Section Parsing & Document Normalization
Raw 10-K and 10-Q filings are processed through [`src/qualitative/html_cleaner.py`](file:///c:/dev/AQQIP/src/qualitative/html_cleaner.py) to strip nested HTML tags, style sheets, and inline XBRL. [`src/qualitative/section_extractor.py`](file:///c:/dev/AQQIP/src/qualitative/section_extractor.py) extracts Item 1A (Risk Factors) and Item 7 (MD&A). If the extracted text contains fewer than 500 words, extraction is marked failed (`extraction_failed`), preventing corrupted text from passing into downstream embeddings.

### 6.2 Embedding Models & Directional Adversity Disentanglement
* **Embeddings:** Evaluated using CPU-compatible sentence-transformers (`all-MiniLM-L6-v2`) and domain-specific regulatory transformers (`nlpaueb/sec-bert-base`).
* **Directional Risk Analyzer ([`src/qualitative/directional_signal.py`](file:///c:/dev/AQQIP/src/qualitative/directional_signal.py)):** Solves the "benign rewrite" failure mode where administrative disclosure changes inflate cosine distance. Uses Loughran-McDonald financial distress dictionaries to identify:
  * *Severe Adverse Patterns:* Going concern, covenant default, restatements, SEC/DOJ subpoenas, insolvency, receivership.
  * *Moderate Adverse Patterns:* Material adverse effect, litigation, credit rating downgrade, supply chain disruption.
  * *Benign Expansion Patterns:* Product expansion, routine geographic updates, commercialization growth.

---

## 7. Multi-Agent Reinforcement Learning (MARL) Market Simulation

### 7.1 Environment Dynamics & Microstructure Invariants
* Custom discrete Gym environment implemented in [`src/marl/engine.py`](file:///c:/dev/AQQIP/src/marl/engine.py).
* **Guaranteed Invariants:**
  1. *Strictly Positive Prices:* Prices are lower-bounded at $\$0.01$; extreme selling pressure cannot produce negative or NaN asset prices.
  2. *Inventory Conservation:* Market maker inventory tracking satisfies $|D_{\text{FP}}| + |D_{\text{TP}}| = |I_{\text{FP}}| + |I_{\text{TP}}|$.
  3. *Finite Returns:* All rewards and drawdowns are filtered through `np.nan_to_num` bounds.

### 7.2 Agent Behavioral Policies
* **Short-Seller:** Scales short exposure based on fused quant probability and directional qualitative adversity, constrained by borrowing liquidity.
* **Retail Herd:** Follows trailing price momentum; panics and sells upon sustained downward price breaks without forward lookahead.
* **Market Maker:** Quotes bilateral spreads, defensively widens spreads under one-sided flow, and adjusts quotes to mean-revert inventory toward zero.

### 7.3 Provenance-Tracked SignalBundle
The bridge module [`src/marl/signal_bridge.py`](file:///c:/dev/AQQIP/src/marl/signal_bridge.py) produces immutable `SignalBundle` objects containing:
* Point-in-time market context (price, volatility, trend).
* Model inference probability, model run ID, and 16-character input feature hash.
* Qualitative adversity scores and filing accession references.
* SHA-256 cryptographic provenance hash guaranteeing auditability.

---

## 8. MLOps Feedback Loop & Prediction Resolution

### 8.1 Resolution Architecture
* [`src/mlops/resolver.py`](file:///c:/dev/AQQIP/src/mlops/resolver.py) reads rows from `predictions` where `status = 'open'` and `target_date <= current_date`.
* It fetches the verified entry price ($P_t$) and exit price ($P_{t+20}$) from MySQL `ohlcv_prices`, computes `actual_price_change_pct = (P_{t+20} - P_t) / P_t * 100.0`, records `actual_drop_observed`, and sets `status = 'resolved'`.
* Idempotency is enforced: re-running resolution over already resolved records produces zero mutations.

### 8.2 Legacy Reconciliation Audit
Historical predictions were audited via [`src/mlops/legacy_reconciliation.py`](file:///c:/dev/AQQIP/src/mlops/legacy_reconciliation.py) against the strict 20-trading-day calendar rule. Non-conforming legacy records were flagged with status `invalid_target` while preserving original timestamps and scores for forensic auditability.

---

## 9. Authoritative Research Metrics & Governance Closure

### 9.1 Phase 2 Asymmetric Volatility Experiment (Closed & Frozen)
* **Governing Specification:** `aqqip_volatility_spec_v4_7.md` (SHA-256: `eabd5e448d038dd26ce5168c4c1bb652d97980e508da53805710d5cfd0d69ec1`).
* **Authoritative Archive:** `C:\dev\AQQIP\experiments\asymmetric_volatility_phase2_20261021_203158`.
* **Formal Decision Document:** [`phase2_closure_addendum.md`](file:///c:/dev/AQQIP/phase2_closure_addendum.md) (SHA-256: `84bf61ee210c624a3d3e9d4cc19b94641fb2f6e01c6ab1e0679ae0e3e06f7523`, 9,916 bytes).
* **Evaluation Panel:** 4 expanding folds (2021, 2022, 2023, 2024), 925 retained dates, 32 equities, 29,600 total observations. Daily alert budget: Top 2 securities ($K_t = 2$, 1,850 total alerts per candidate).

#### Pooled Performance Scorecard (Audited Machine Values)
| Candidate | Role | Features | TP | FP | TN | FN | False-Positive Share | Recall | Precision | F1 Score | Selected Rounds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Candidate A** | **Baseline (Audited Production Standard)** | 11 | 466 | 1,384 | 24,154 | 3,596 | **74.8108%** | **11.4722%** | 25.1892% | 0.157645 | 46.75 |
| **Candidate B** | Exploratory (Additive DSV) | 12 | 461 | 1,389 | 24,149 | 3,601 | 75.0811% | 11.3491% | 24.9189% | 0.155954 | 45.50 |
| **Candidate C** | Primary Challenger (Substituted DSV) | 11 | 452 | 1,398 | 24,140 | 3,610 | **75.5676%** | **11.1275%** | 24.4324% | 0.152909 | 49.75 |
| **Candidate D** | Exploratory (Additive DAR) | 12 | 471 | 1,379 | 24,159 | 3,591 | 74.5405% | 11.5953% | 25.4595% | 0.159337 | 50.00 |

#### Primary Statistical Inference & Formal Hypothesis Test
* **False-Positive Share Difference (C minus A):** $\hat{\Delta} = +0.7568\text{ percentage points}$ ($+14$ net false positives).
* **Two-Sided 95% Bootstrap Confidence Interval:** $[-0.2162, +1.8378]\text{ percentage points}$.
  * *Statistical Interpretation:* The confidence interval spans zero ($0.0\text{ pp}$). The difference does **not** establish statistically significant deterioration at $\alpha = 0.05$.
  * *Hurdle Failure:* Candidate C decisively failed both registered promotion criteria: required target effect size ($\le -4.0\text{ pp}$) and statistical detectability (upper bound $< 0.0\text{ pp}$).
* **Recall Non-Inferiority:** 5th percentile bootstrap lower bound = $-0.7955\text{ pp} \ge -1.0\text{ pp}$ hurdle (**PASS**).
* **Mechanistic Attribution:** $74.18\%$ of displaced false positives ($158 / 213$) occurred on stocks with positive forward returns (**PASS**, $> 60.0\%$).
* **Authoritative Governance Decision:** **Outcome O-8B** (Tier 3: Baseline A retained; Candidate C rejected for deployment). The 2021–2024 evaluation panel is **closed and frozen**.

### 9.2 Diagnostic Holdout Models (2025–2026 Panel)
Evaluated on [`eval_artifacts/authoritative_eval_predictions_2025_2026.parquet`](file:///c:/dev/AQQIP/eval_artifacts/authoritative_eval_predictions_2025_2026.parquet) from MLflow:
* **Model M (`59a7f799...`):** ROC-AUC = 0.6170, PR-AUC = 0.1916, Brier = 0.11477, F1@op = 0.2738.
* **Model MF (`31ac56b4...`):** ROC-AUC = 0.6077, PR-AUC = 0.1805, Brier = 0.11549, F1@op = 0.2713.
* **Model MFQ (`54a2403a...`):** ROC-AUC = 0.5775, PR-AUC = 0.1590, Brier = 0.11641, F1@op = 0.2168.
* **Model M_Regularized (`62d5c881...`):** ROC-AUC = 0.6165, PR-AUC = 0.1878, Brier = 0.11447, F1@op = 0.2780, Recall@op = 0.7359.
* **Production Baseline Model (`df161d0b...`):** test ROC-AUC = 0.5871, test PR-AUC = 0.1723, test Brier = 0.0852. Default threshold $0.50$ produced 0.0 precision/recall because event base rate is $\approx 9.5\%$, validating the necessity of calibrated operational thresholds ($p \ge 0.1600$).

### 9.3 Phase 3 Research Design Audit (v1.1 vs. v1.2)
* **Authoritative Design Document:** [`phase3_research_design_v1_2.md`](file:///c:/dev/AQQIP/phase3_research_design_v1_2.md) (SHA-256: `32056b50b86ec7af7e126fddee249dd981747c96ea401ba821abc0fbb99bee65`, 21,090 bytes).
* **Research Core:** Evaluating whether replacing point-wise binary log-loss with session-grouped list-wise ranking (LambdaMART NDCG@2) reduces operational top-2 false-positive share.
* **Methodological Audit of v1.1:** Version 1.1 claimed 84.8% power to detect a $-4.0\text{ pp}$ effect by testing $H_0: \mu \ge 0$. Version 1.2 proved this claim defective because the decision rule requires the sample point estimate to satisfy $\hat{\Delta} \le -4.0\text{ pp}$; under true $\mu = -4.0\text{ pp}$, the power to exceed that threshold is capped at approximately $50\%$.
* **Governance Boundary:** All Phase 3 exploratory tuning is quarantined to pre-2021 data (2014–2020). The 2021–2024 and 2025–2026 panels are strictly closed to prevent data snooping.

---

## 10. Frontend Architecture, Pending Integration & Visual Constraints

### 10.1 Current Implementation State
The existing frontend in [`dashboard/app.py`](file:///c:/dev/AQQIP/dashboard/app.py) is a minimal, 32-line Streamlit placeholder designed solely to verify container startup and Docker healthcheck routing. **Zero business views, chart components, or data-binding pages have been implemented yet.**

### 10.2 Planned Architecture (Streamlit-Native)
Per [Sprint 6 Plan](file:///c:/dev/AQQIP/plan.md#L88-L98), the dashboard is designed as a read-only Streamlit interface with 3 primary views:
1. **Live Accuracy Tracker:** Rolling 30/60/90-day hit rate, MAE, and Brier skill score read directly from MySQL `predictions`.
2. **Lie Spotter:** Side-by-side disclosure diff viewer highlighting deletions and additions between consecutive 10-Ks from MongoDB `semantic_diffs`.
3. **MARL Crash Simulation:** Animated microstructure price paths and order book liquidity depletion charts from MySQL `marl_simulation_runs`.

### 10.3 Visual Design References & Strict Anti-Patterns
The user provided two design reference links for the forthcoming frontend implementation:
* **Positive Reference:** `https://www.instagram.com/reel/De7aochhik3/` (interaction design, data density, information hierarchy).
* **Anti-Reference:** `https://www.instagram.com/reel/DcrqbqZRVvC/` (low-quality AI/SaaS design tropes).

#### The 20 Forbidden Design Patterns (Mandatory Constraints)
The user supplied an explicit list of 20 patterns that **must NOT be introduced** into the frontend:
1. Purple-to-blue gradients
2. Gradient hero text
3. Emojis in headings
4. Inter font everywhere
5. Colored-border cards
6. Glassmorphism cards
7. Low-contrast dark mode
8. Three icon boxes in a row
9. A badge above the headline
10. Lucide icons everywhere
11. Untouched default shadcn UI
12. Fade-in-on-scroll animations
13. Cursor-following beam
14. Buttons fading on hover
15. Inconsistent spacing
16. Em dashes everywhere
17. Generic buzzword copy
18. Serif italic accents
19. Space Grotesk + Instrument Serif font pairing
20. Grain texture over a gradient

*Note:* Automated tools cannot access Instagram reels directly. When frontend implementation begins, the user should provide screenshots or screen recordings of the positive reference if specific visual structures are to be mirrored.

---

## 11. Known Engineering Milestones & Historical Context

1. **Phase 1 Infrastructure Exit (Verified):** Standup of MySQL 8.0, MongoDB 7.0, MLflow, and ChromaDB persistence in Docker Compose. Initial test suite verified 104 passing tests synchronously.
2. **Phase 2 Ingestion & Data Readiness (Verified):** Integration of `yfinance` daily pricing and SEC EDGAR full-text fetching. Ingestion idempotency proven with unique key constraints.
3. **Phase 2 Asymmetric Volatility Protocol (Audited & Closed):** Formal execution of the 4-fold rolling-origin experiment (Invocation 2, run `20261021_203158`). Registered Outcome O-8B retained Baseline A.
4. **Target Date Reconciliation Audit (Verified):** Audited MySQL prediction records against the exact 20-trading-day forward contract, purging legacy discrepancies.
5. **Phase 3 Research Design Audit (Verified):** Corrected statistical power claims in v1.1; published authoritative design specification v1.2.

---

## 12. Actionable Status Register and Prioritized Roadmap

| Priority | Workstream | Specific Task | Status | Relevant Files | Primary Risk / Acceptance Criteria |
|---|---|---|---|---|---|
| **P1** | **Research Governance** | Maintain freeze on Phase 2 (2021–2024) and holdout (2025–2026) panels | **Active / Enforced** | [`phase2_closure_addendum.md`](file:///c:/dev/AQQIP/phase2_closure_addendum.md) | Prevent retrospective data-snooping or p-hacking. |
| **P2** | **Frontend UI** | Implement 3 Streamlit business views (Accuracy Tracker, Lie Spotter, MARL Crash) | **Pending Implementation** | [`dashboard/app.py`](file:///c:/dev/AQQIP/dashboard/app.py), `dashboard/views/` | Must query live MySQL/Mongo data; strictly adhere to the 20 anti-pattern restrictions. |
| **P3** | **Testing / Environment**| Resolve missing packages (`sentence_transformers`, `langchain_core`) on host Python 3.14 | **Incomplete on Host** | `requirements.txt`, Dockerfile.python | Tests pass in Docker Python 3.11 but fail collection on host. Requires venv alignment. |
| **P4** | **Phase 3 Prospective Study**| Build prospective execution harness for LambdaMART NDCG@2 challenger | **Designed (v1.2)** | [`phase3_research_design_v1_2.md`](file:///c:/dev/AQQIP/phase3_research_design_v1_2.md) | Restrict model fitting to pre-2021 data; prospective evaluation over $\ge 500$ live sessions. |
| **P5** | **MLOps Automation** | Wire cron supervisor to execute `resolver.py` daily on production schedules | **Partially Automated** | [`src/mlops/resolver.py`](file:///c:/dev/AQQIP/src/mlops/resolver.py), [`docker-compose.yml`](file:///c:/dev/AQQIP/docker-compose.yml) | Ensure scheduled resolver runs autonomously in background without manual invocation. |

---

## 13. Unresolved Questions, Contradictions, and Governance Risks

1. **Statistical Power Disconnect in Prospective Design:** As audited in Section 9.3 and documented in [`phase3_research_design_v1_2.md`](file:///c:/dev/AQQIP/phase3_research_design_v1_2.md), previous claims of $>80\%$ power to detect a $-4.0\text{ pp}$ improvement under joint criteria remain invalid. Offline pre-2021 simulation must establish true effect-size variance before finalizing prospective sample sizes.
2. **Host Python 3.14 vs. Docker Python 3.11 Divergence:** Host environment lacks `sentence_transformers` and `langchain_core`, causing test collection errors when running `pytest` directly on the host. Integration tests must be executed inside Docker or a dedicated host Python 3.11 virtualenv.
3. **Execution Summary Runner Hash Gap:** `execution_summary.json` omitted the SHA-256 hash of `run_asymmetric_volatility_experiment.py`. Future execution harnesses must automatically inject `__file__` SHA-256 directly into the summary JSON.
4. **Bootstrap Block Resampling Sequences:** While bootstrap quantile outputs are fully reproducible using random seed `20240101`, individual resampled indices were not persisted to disk. Future protocols should persist the draw index table.

---

## 14. COPY THIS INTO THE NEXT CHATGPT CONVERSATION

```markdown
You are an expert quantitative researcher, forensic financial analyst, and MLOps software engineer continuing development on the AQQIP project.

### CORE PROJECT IDENTITY & RULES:
- Project: AQQIP (Autonomous Quantitative & Qualitative Investment Platform)
- Repository Root: C:\dev\AQQIP
- Primary Context File: docs/handover/AQQIP_PROJECT_HANDOVER.md
- Machine-Readable Metrics: docs/handover/AQQIP_METRICS_SNAPSHOT.json
- Directory Tree: docs/handover/AQQIP_REPOSITORY_TREE.txt
- Hardware & Budget Constraints: Runs locally on NVIDIA RTX 4060 Laptop (8GB VRAM), 16GB RAM, AMD Ryzen 7 7840HS. Hard $0 budget (no paid cloud infra or paid APIs; only Gemini Free Tier permitted; all embeddings CPU-based).

### CRITICAL RESEARCH GOVERNANCE BOUNDARIES (DO NOT VIOLATE):
1. Phase 2 Evaluation (2021–2024 panel) is CLOSED and FROZEN under Outcome O-8B. Baseline A remains the authoritative production model (FP Share = 74.8108%, Recall = 11.4722%). Candidate C (substituted Downside Semivolatility) was rejected.
2. The 2021–2024 panel and the 2025–2026 diagnostic panel must NEVER be reloaded, rescored, retuned against, or used for model selection.
3. Phase 3 challenger research (LambdaMART NDCG@2 ranking) must restrict all hyperparameter tuning and model selection exclusively to pre-2021 data (2014–2020), following phase3_research_design_v1_2.md.
4. Do NOT rerun closed-panel experiments, refit models, or modify frozen files in manifests/ or eval_artifacts/.

### IMMEDIATE PENDING WORK:
1. Frontend Dashboard Implementation: The dashboard (dashboard/app.py) is currently a minimal container stub. We need to implement the 3 real read-only Streamlit views:
   - Live Accuracy Tracker (reads MySQL `predictions` table)
   - Lie Spotter (side-by-side SEC filing diff viewer reading MongoDB `semantic_diffs`)
   - MARL Crash Simulation (animated drawdown and liquidity depletion charts reading MySQL `marl_simulation_runs`)
2. Visual Constraints: The user provided a positive visual reference (Instagram reel De7aochhik3) and an explicit anti-reference (Instagram reel DcrqbqZRVvC). Avoid all 20 listed anti-patterns:
   - No purple-to-blue gradients, gradient hero text, emojis in headings, Inter font everywhere, colored-border cards, glassmorphism, low-contrast dark mode, 3 icon boxes in a row, badges above headlines, Lucide icons everywhere, untouched default shadcn, fade-in-on-scroll animations, cursor beams, button hover fades, inconsistent spacing, em dashes, generic buzzwords, serif italic accents, Space Grotesk + Instrument Serif pairings, or grain over gradients.
3. Data Integrity: All dashboard views must query real live MySQL and MongoDB databases. NO hardcoded, fabricated, or mock financial data.

Before proposing code, inspect the actual target files and verify existing implementations. Do not speculate or modify frozen research artifacts.
```
