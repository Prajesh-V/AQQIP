# Product Requirements Document (PRD)
## Autonomous Quantitative & Qualitative Investment Platform (AQQIP)

**Document Version:** 1.0
**Status:** Draft — Foundational
**Cost Constraint:** $0 (no paid cloud infra, no paid APIs)

---

## 1. Executive Summary

AQQIP is a B2B forensic audit and adversarial short-seller simulation engine. It functions as an "Analyst Force Multiplier" — a system that ingests public financial data and SEC filings, applies quantitative anomaly detection, semantic drift analysis, and multi-agent reinforcement learning (MARL) to flag potential corporate fraud and simulate the market impact of a short thesis before it becomes consensus. The system continuously grades its own predictions against real market outcomes and retrains itself (self-correcting MLOps loop).

This document defines what AQQIP must do, for whom, how success is measured, and — critically — what it explicitly does **not** do.

---

## 2. Target Audience / Personas

| Persona | Description | Primary Need |
|---|---|---|
| **Forensic Analyst** | Buy-side/sell-side research analyst screening for accounting fraud | Fast triage of a large ticker universe down to a short-list of anomalous names |
| **Independent Short-Seller / Research Boutique** | Small team producing public short theses | Evidence-backed anomaly detection + simulated market reaction to size a position thesis |
| **Compliance / Internal Audit Team** | Corporate risk function | Early warning on subsidiaries or portfolio holdings exhibiting red-flag patterns |
| **Quant Researcher / MLOps Engineer** | Builds and maintains the models | Transparent, versioned, locally reproducible pipeline with full experiment tracking |

AQQIP is **not** designed for retail investors seeking trade signals, and it is **not** a brokerage or execution platform.

---

## 3. Problem Statement

Traditional fraud detection is manual, slow, and reactive — analysts read hundreds of pages of 10-K/10-Q filings by hand, looking for language changes or ratio anomalies that often only become obvious in hindsight (post-restatement, post-crash). Meanwhile, understanding what a short thesis is *worth* requires modeling not just "is this company lying" but "what happens to the stock if the market agrees with me." AQQIP unifies these two normally-separate workflows — accounting forensics and market microstructure simulation — into a single, self-improving, zero-cost pipeline.

---

## 4. User Stories

### Data & Ingestion
- As an analyst, I want the system to automatically pull daily OHLCV price history for my watchlist so I don't have to manually download CSVs.
- As an analyst, I want the system to automatically poll SEC EDGAR for new 10-K/10-Q filings for tracked tickers so nothing is missed.

### Quantitative Engine
- As an analyst, I want to see the Beneish M-Score and Altman Z-Score for a ticker over time so I can spot deteriorating fundamentals.
- As a quant researcher, I want the LightGBM anomaly model's hyperparameters tuned automatically (Optuna) and every run logged (MLflow) so results are reproducible and comparable across experiments.

### Qualitative Engine
- As an analyst, I want the system to diff "Risk Factors" language between consecutive filing years so I can see exactly what language was softened, added, or deleted.
- As an analyst, I want a semantic similarity score between Year 1 and Year 2 filings so I can quantify how much the disclosure tone shifted, not just eyeball it.

### MARL Simulation
- As a short-seller, I want to simulate how a short-seller/retail-herd/market-maker interaction plays out given a liquidity profile, so I can estimate plausible drawdown and expected profit before committing capital.

### MLOps Loop
- As a quant researcher, I want every prediction logged with its target date and actual outcome so the system's real-world hit rate is always visible and auditable.
- As a quant researcher, I want the model to automatically retrain on hard negatives (false positives/negatives) on a schedule, without manual intervention.

### Dashboard
- As an analyst, I want a single Streamlit dashboard showing live accuracy, the "Lie Spotter" text diff, and the MARL simulation output for any ticker I select.

---

## 5. Definition of Done (DoD) — Strict

A feature/module is only considered "Done" when **all** of the following hold:

1. **Runs fully locally** via `docker compose up` with zero paid service dependencies (Gemini Free Tier is the sole external LLM call; all other components run in-container or embedded).
2. **Idempotent ingestion** — re-running the ingestion pipeline does not create duplicate rows/documents (enforced via upserts / unique constraints).
3. **Every model training run is logged to MLflow** with parameters, metrics, and artifact (model binary) versioned.
4. **Every prediction is persisted** to MySQL with a `target_date` and is later resolved (accurate/inaccurate) by the daily evaluation cron — no orphaned predictions.
5. **Unit test coverage** exists for all data-transformation and scoring functions (Beneish, Altman, MARL reward functions).
6. **Rate limits are respected** — all external calls (yfinance, SEC EDGAR, Gemini) use retry/backoff (Tenacity) and stay within documented free-tier limits.
7. **Dashboard reflects live DB state** — no hardcoded/mock data in Streamlit views.
8. **Documented** — any new module ships with an update to `architecture.md` if it changes a data flow or schema.

---

## 6. Success Metrics

| Category | Metric | Target |
|---|---|---|
| **Quant Model Quality** | Precision @ top-decile anomaly score (does a flagged stock actually see a sharp drop within the prediction horizon) | ≥ 60% precision on backtest holdout |
| **Quant Model Quality** | AUC-ROC of LightGBM drop-probability model | ≥ 0.70 |
| **Qualitative Engine** | Manual-review agreement rate (does the "Lie Spotter" flag correlate with human-identified material disclosure changes on a labeled sample set) | ≥ 75% agreement on a 50-filing sample |
| **MLOps Loop** | Live rolling hit-rate (30/60/90-day) visible on dashboard, computed without manual query | 100% automated, refreshed daily |
| **System Latency** | End-to-end time to fully process one new ticker (ingest → score → RAG index → dashboard-ready) | < 5 minutes on standard dev laptop (CPU-only) |
| **System Latency** | MARL simulation run time (single scenario, default agent count) | < 30 seconds |
| **Reliability** | Ingestion pipeline success rate over a rolling 30-day window (excluding external API outages) | ≥ 98% |
| **Cost** | Total infrastructure + API spend | $0.00 (hard constraint, not aspirational) |

---

## 7. Out of Scope (Explicit Non-Goals)

The following are **explicitly excluded** from this project. Any future work in these areas requires a separate PRD and is not to be silently added during implementation:

- **No real-money live trading or order execution.** AQQIP produces research signals and simulations only; it never connects to a brokerage API or places real orders.
- **No paid cloud infrastructure** (AWS/GCP/Azure) — everything must run in local Docker containers.
- **No paid LLM APIs** beyond the Google Gemini Free Tier (via Google AI Studio / LangChain).
- **No real-time/tick-level market data.** The system uses daily OHLCV from yfinance, not a low-latency feed.
- **No coverage of non-US filers / non-SEC-EDGAR jurisdictions** in v1 (SEBI SLB mechanics are used only as the *simulation model* for short-selling constraints, not as a live India-market data integration).
- **No legal or investment advice.** All outputs are research/simulation artifacts; the system does not generate buy/sell recommendations framed as advice, and all dashboard outputs must carry a "research tool, not investment advice" disclaimer.
- **No multi-tenant SaaS / user auth / billing system** in this phase — this is a single-operator local deployment.
- **No custom frontend design system.** The Streamlit dashboard uses default/functional styling only; polished UI/UX is explicitly deferred to a future cloned template (see `memory.md`).
- **No mobile app or browser extension.**
- **No support for unstructured filing types beyond 10-K/10-Q** (e.g., no 8-K, S-1, or proxy statement parsing in v1).

---

## 8. Assumptions & Constraints

- Developer has Docker & Docker Compose installed locally; no orchestration (Kubernetes) is used.
- Google AI Studio Free Tier API key is available and its request-per-minute limits are respected via Tenacity-based backoff.
- yfinance and SEC EDGAR's public REST API remain free and accessible without authentication (EDGAR requires only a compliant `User-Agent` header, no key).
- All persistent state (MySQL, MongoDB, MLflow's SQLite backend, ChromaDB's local persistence) is stored on local Docker volumes — no cloud storage.
- The system is designed for a single operator / single team, not concurrent multi-user production load.
