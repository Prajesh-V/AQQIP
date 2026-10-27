# Development Phases
## Autonomous Quantitative & Qualitative Investment Platform (AQQIP)

Each phase is **linear, independent, and testable** — it must produce a working, verifiable slice of the system before the next phase begins. No phase should require code from a *later* phase to be considered complete.

---

## Phase 1 — Environment & Databases

**Goal:** Stand up the full local infrastructure with zero application logic.

**Scope:**
- `docker-compose.yml` defining `mysql`, `mongodb`, and placeholder service stubs for later phases.
- MySQL schema migration scripts (`ohlcv_prices`, `fundamental_ratios`, `predictions`, `marl_simulation_runs` tables created).
- MongoDB collections initialized with indexes (`sec_filings`, `semantic_diffs`).
- `.env.example` documenting all required environment variables (DB credentials, Gemini API key placeholder).
- ChromaDB local persistence directory verified writable.

**Test / Exit Criteria:**
- `docker compose up` brings up all containers healthy.
- Can connect to MySQL and MongoDB from host via CLI client and confirm empty tables/collections exist with correct schema.
- No Python business logic exists yet — this phase is infrastructure-only.

---

## Phase 2 — Data Ingestion Pipeline

**Goal:** Reliably pull market prices and SEC filings into the databases.

**Scope:**
- `yfinance_fetcher.py`: pulls historical OHLCV for a configurable ticker list, upserts into `ohlcv_prices`.
- `edgar_fetcher.py`: polls SEC EDGAR full-text search / submissions API for 10-K/10-Q filings, stores raw filing text + metadata into `sec_filings`.
- Tenacity-based retry/backoff wrapper for both fetchers.
- Basic scheduler (APScheduler or cron) to run ingestion on a defined interval.
- Unit tests for upsert idempotency (re-running ingestion does not duplicate rows).

**Test / Exit Criteria:**
- Running the ingestion pipeline twice in a row for the same ticker/date range results in no duplicate rows in MySQL (verified by `UNIQUE` constraint + row count check).
- At least 3 real tickers' worth of price history and at least 1 real 10-K filing per ticker successfully ingested end-to-end.
- Rate-limit handling verified by simulating a 429/503 response and confirming retry behavior.

---

## Phase 3 — Quantitative Engine (Math Fraud Detection)

**Goal:** Compute forensic accounting ratios and train the anomaly-prediction model.

**Scope:**
- `beneish_score.py` / `altman_score.py`: compute M-Score and Z-Score from ingested fundamentals, write to `fundamental_ratios`.
- `lightgbm_trainer.py`: trains a LightGBM classifier predicting probability of sharp price drop, using ratio features + price history as labels.
- Optuna integration for hyperparameter tuning.
- MLflow integration: every training run logs params, metrics (AUC, precision/recall), and the model artifact.
- Prediction persistence: model inference results written to `predictions` table with `target_date`.

**Test / Exit Criteria:**
- MLflow UI (`localhost:5000`) shows at least one completed run with logged metrics and a downloadable model artifact.
- A backtest script demonstrates the model produces a probability score for a holdout set, with AUC-ROC computed and printed.
- `predictions` table contains rows with `status='open'` for the current ticker universe.

---

## Phase 4 — Qualitative Engine (Semantic Shift RAG)

**Goal:** Detect and summarize deceptive language shifts between filing years.

**Scope:**
- `nltk_preprocessor.py`: extracts and tokenizes the "Risk Factors" section from raw filing text.
- `embedding_service.py`: chunks and embeds text via `sentence-transformers/all-MiniLM-L6-v2`, upserts into ChromaDB with chronological metadata.
- `semantic_diff_agent.py`: LangChain agent (Gemini-backed) that retrieves Year N vs. Year N-1 chunks, computes cosine similarity, identifies deleted/added/shifted sentences, and generates a natural-language interpretation.
- Results persisted to MongoDB `semantic_diffs` collection.

**Test / Exit Criteria:**
- For at least one ticker with two consecutive years of 10-K filings ingested, a `semantic_diffs` document is produced containing non-empty `deleted_sentences`/`added_sentences` arrays and a Gemini-generated `llm_summary`.
- Manual spot-check confirms the diff correctly identifies at least one genuine textual change between the two filings.
- Embedding step confirmed CPU-only (no GPU dependency) and completes within a reasonable time budget (<2 min for one filing).

---

## Phase 5 — MARL Simulation Engine & MLOps Loop

**Goal:** Simulate market reaction to a short thesis and close the self-correction loop.

**Scope:**
- Custom Gym environment (`marl_env.py`) implementing three agent types: Short-Seller, Retail Herd, Market Maker, under SEBI SLB-inspired short-selling constraints.
- `marl_runner.py`: runs a simulation seeded with the quant anomaly score + qualitative shift score, outputs max drawdown and expected short profit to `marl_simulation_runs`.
- `mlops_daily_job.py`: daily cron job that queries `predictions` for rows past `target_date` with `status='open'`, pulls current price via yfinance, resolves `actual_drop_observed`, updates row to `status='resolved'`.
- Feedback loop: resolved hard-negative/hard-positive examples fed into the next scheduled LightGBM retrain (Phase 3 trainer re-invoked with updated training set); MARL reward penalty coefficients adjusted based on rolling accuracy.

**Test / Exit Criteria:**
- A full MARL simulation run completes in under 30 seconds and produces a plausible (bounded, non-NaN) drawdown/profit output.
- The daily MLOps job, run manually against a `predictions` row with a `target_date` in the past, correctly resolves it and updates the row.
- A second LightGBM training run (post-feedback) is visible in MLflow with a different `run_id`, demonstrating the retrain loop executes.

---

## Phase 6 — Streamlit Visualization Dashboard

**Goal:** Unify all outputs into a single, live, read-only dashboard.

**Scope:**
- `dashboard_app.py` with three views:
  1. **Live Accuracy Tracker** — hit rate and MAE computed live from `predictions` table.
  2. **Lie Spotter** — renders the diff (deletions/additions highlighted) from `semantic_diffs` + raw chunk text.
  3. **MARL Crash Simulation** — animated chart of the simulated price path / drawdown from the latest `marl_simulation_runs` entry.
- All views are read-only against the live databases — no hardcoded/mock data permitted (per DoD in `prd.md`).
- Ticker selector driving all three views from a single dropdown.

**Test / Exit Criteria:**
- Dashboard launches at `localhost:8501` and all three views render real data for at least one fully-processed ticker (i.e., one that has passed through Phases 2–5).
- Changing the ticker selector updates all three views without error.
- Confirmed no mock/placeholder data remains in the final dashboard code.

---

## Phase Dependency Summary

```
Phase 1 (Infra)
    │
    ▼
Phase 2 (Ingestion) ──────────────┐
    │                              │
    ▼                              ▼
Phase 3 (Quant Engine)     Phase 4 (Qualitative Engine)
    │                              │
    └──────────────┬───────────────┘
                    ▼
           Phase 5 (MARL + MLOps Loop)
                    │
                    ▼
           Phase 6 (Dashboard)
```

Phases 3 and 4 may be developed in parallel once Phase 2 is complete, since they consume different data stores (MySQL vs. MongoDB/ChromaDB) and do not depend on each other's outputs — but Phase 5 requires both to be complete, and Phase 6 requires Phase 5.
