# Actionable Sprint Plan
## Autonomous Quantitative & Qualitative Investment Platform (AQQIP)

Granular, file-by-file checklist. Mirrors `phases.md` 1:1. Check items off in order — each item should be a small, committable unit of work. No Python code is included here, only file paths and their responsibilities.

---

## Sprint 1 → Phase 1: Environment & Databases

- [ ] `docker-compose.yml` — define `mysql`, `mongodb` services, shared `aqqip-net` network, named volumes
- [ ] `.env.example` — document `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE`, `MONGO_INITDB_DATABASE`, `GEMINI_API_KEY`, `EDGAR_USER_AGENT`
- [ ] `db/mysql/init/001_create_ohlcv_prices.sql`
- [ ] `db/mysql/init/002_create_fundamental_ratios.sql`
- [ ] `db/mysql/init/003_create_predictions.sql`
- [ ] `db/mysql/init/004_create_marl_simulation_runs.sql`
- [ ] `db/mongo/init/init-indexes.js` — creates `sec_filings` and `semantic_diffs` collections with indexes
- [ ] `config/settings.py` — centralized config loader (reads `.env`, exposes typed config object)
- [ ] `README.md` — quickstart: `docker compose up`, verify containers healthy
- [ ] **Test:** `tests/test_infra_healthcheck.py` — connects to MySQL and MongoDB, asserts empty tables/collections exist

---

## Sprint 2 → Phase 2: Data Ingestion Pipeline

- [ ] `src/database/mysql_client.py` — connection pool + upsert helper functions
- [ ] `src/database/mongo_client.py` — connection helper + upsert-by-`accession_number` helper
- [ ] `src/ingestion/yfinance_fetcher.py` — pulls OHLCV for ticker list, calls `mysql_client` upsert
- [ ] `src/ingestion/edgar_fetcher.py` — polls SEC EDGAR submissions API, filters 10-K/10-Q, calls `mongo_client` upsert
- [ ] `src/ingestion/section_extractor.py` — stub for later NLTK use (raw text passthrough for now)
- [ ] `src/utils/retry.py` — Tenacity decorator wrappers for HTTP calls
- [ ] `src/ingestion/scheduler.py` — APScheduler/cron entrypoint for periodic ingestion
- [ ] `src/ingestion/ticker_universe.py` — config-driven list of tracked tickers
- [ ] **Test:** `tests/test_yfinance_fetcher_idempotency.py`
- [ ] **Test:** `tests/test_edgar_fetcher_idempotency.py`
- [ ] **Test:** `tests/test_retry_backoff.py` — mocks a 429 response, asserts retry occurs

---

## Sprint 3 → Phase 3: Quantitative Engine

- [ ] `src/quant/beneish_score.py` — Beneish M-Score calculation from fundamentals
- [ ] `src/quant/altman_score.py` — Altman Z-Score calculation
- [ ] `src/quant/feature_builder.py` — assembles feature matrix (ratios + price-derived features) for model input
- [ ] `src/quant/lightgbm_trainer.py` — trains classifier, integrates Optuna study
- [ ] `src/quant/optuna_search_space.py` — defines hyperparameter search space
- [ ] `src/quant/mlflow_tracking.py` — wraps training run in MLflow `start_run`, logs params/metrics/artifact
- [ ] `src/quant/predictor.py` — loads latest MLflow-registered model, writes new rows to `predictions`
- [ ] `src/quant/backtest.py` — holdout evaluation script, prints AUC/precision/recall
- [ ] **Test:** `tests/test_beneish_score.py` — known-input/known-output cases
- [ ] **Test:** `tests/test_altman_score.py`
- [ ] **Test:** `tests/test_predictor_writes_open_prediction.py`

---

## Sprint 4 → Phase 4: Qualitative Engine

- [ ] `src/qualitative/nltk_preprocessor.py` — sentence tokenization, "Risk Factors" section extraction (regex/heading-based)
- [ ] `src/qualitative/chunker.py` — paragraph-level chunking (~200–400 tokens)
- [ ] `src/qualitative/embedding_service.py` — HuggingFace `all-MiniLM-L6-v2` embedding wrapper (CPU)
- [ ] `src/qualitative/chroma_client.py` — ChromaDB local client, per-ticker collection management, metadata-filtered queries
- [ ] `src/qualitative/similarity_diff.py` — pairwise cosine similarity between Year N and Year N-1 chunks, deleted/added/shifted classification
- [ ] `src/qualitative/semantic_diff_agent.py` — LangChain agent wrapping Gemini API call to produce `llm_summary`
- [ ] `src/qualitative/diff_persistence.py` — writes result to MongoDB `semantic_diffs`
- [ ] **Test:** `tests/test_chunker.py`
- [ ] **Test:** `tests/test_similarity_diff_known_case.py` — synthetic before/after text pair with known expected diff
- [ ] **Test:** `tests/test_chroma_client_metadata_filter.py`

---

## Sprint 5 → Phase 5: MARL Simulation Engine & MLOps Loop

- [ ] `src/marl/gym_env.py` — custom `gym.Env` subclass defining state/action/reward spaces
- [ ] `src/marl/agents/short_seller_agent.py`
- [ ] `src/marl/agents/retail_herd_agent.py`
- [ ] `src/marl/agents/market_maker_agent.py`
- [ ] `src/marl/reward_functions.py` — reward shaping, including penalty coefficients adjustable by MLOps loop
- [ ] `src/marl/marl_runner.py` — orchestrates a simulation run, seeds from quant + qualitative scores, writes to `marl_simulation_runs`
- [ ] `src/mlops/daily_resolver.py` — queries open predictions past `target_date`, resolves via yfinance current price
- [ ] `src/mlops/feedback_loop.py` — assembles hard-negative/hard-positive training set, triggers `lightgbm_trainer.py` retrain
- [ ] `src/mlops/reward_adjuster.py` — updates MARL reward penalty coefficients based on rolling accuracy from `predictions`
- [ ] `src/mlops/cron_entrypoint.py` — daily job entrypoint (calls resolver → feedback_loop → reward_adjuster in sequence)
- [ ] **Test:** `tests/test_marl_runner_bounded_output.py` — asserts drawdown/profit outputs are finite and within plausible bounds
- [ ] **Test:** `tests/test_daily_resolver_resolves_past_prediction.py`
- [ ] **Test:** `tests/test_reward_adjuster_updates_coefficients.py`

---

## Sprint 6 → Phase 6: Streamlit Visualization Dashboard

- [ ] `dashboard/app.py` — main Streamlit entrypoint, ticker selector, page routing
- [ ] `dashboard/views/accuracy_tracker.py` — live hit-rate/MAE view (reads `predictions`)
- [ ] `dashboard/views/lie_spotter.py` — diff-highlighting view (reads `semantic_diffs` + Mongo raw text)
- [ ] `dashboard/views/marl_crash_view.py` — animated simulation chart (reads `marl_simulation_runs`)
- [ ] `dashboard/data_access.py` — read-only query layer shared across views (no writes from dashboard)
- [ ] `dashboard/disclaimers.py` — renders the mandatory "research tool, not investment advice" banner
- [ ] **Test:** `tests/test_dashboard_data_access_no_mock_data.py` — asserts all queries hit live DB, not fixtures
- [ ] **Manual QA:** launch dashboard, switch tickers, confirm all three views update correctly

---

## Cross-Cutting Files (created incrementally as needed, not tied to one sprint)

- [ ] `requirements.txt` — pinned versions for all Python dependencies
- [ ] `Dockerfile.python` — shared base image for all Python services
- [ ] `memory.md` — AI coding assistant rulebook (see companion document)
- [ ] `.github/CONTRIBUTING.md` *(optional, if repo is shared)*
- [ ] `logs/` — structured logging output directory (gitignored)
