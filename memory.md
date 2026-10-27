# AI Coding Assistant Rulebook
## Autonomous Quantitative & Qualitative Investment Platform (AQQIP)

This document is the persistent memory/ruleset for any AI coding agent (Claude Code, Copilot, Cursor, etc.) working on this repository. Read this before generating or modifying any code. These rules are **non-negotiable** unless explicitly overridden by a human maintainer in writing.

---

## 1. Cost Constraints — Hard Rules

- **Never introduce a paid API or paid cloud service.** No AWS, GCP, Azure, Snowflake, Pinecone (paid tier), OpenAI API, Anthropic API, or any service requiring a credit card.
- The **only** external LLM call permitted is to the **Google Gemini API via the free tier** (Google AI Studio key), accessed through LangChain.
- All databases (MySQL, MongoDB, ChromaDB) must run **locally in Docker** or as embedded/local processes. No managed database services.
- MLflow uses its **local SQLite backend** (`./mlruns/`) — never configure a remote tracking server or S3-backed artifact store.
- If a task seems to require a paid service, **stop and flag it** rather than substituting one silently.

---

## 2. Port Reference (Do Not Reassign Without Updating This File)

| Service | Port |
|---|---|
| MySQL | `3306` |
| MongoDB | `27017` |
| Streamlit Dashboard | `8501` |
| MLflow UI | `5000` |
| ChromaDB | embedded (no port — local persistence directory, not a server) |

---

## 3. Code Quality Rules

- **Always use type hints** on every function signature (parameters and return types). No exceptions for "quick scripts" — this is a research/production hybrid system and untyped code is not acceptable.
- **Always handle external-call rate limits with Tenacity.** Any call to `yfinance`, SEC EDGAR REST endpoints, or the Gemini API must be wrapped in a Tenacity retry decorator with exponential backoff. Never call these APIs "bare."
- **Idempotency is mandatory for all ingestion code.** Every insert into MySQL or MongoDB must be an upsert keyed on a natural unique key (e.g., `(ticker, trade_date)` for prices, `accession_number` for filings). Re-running any ingestion script must never create duplicates.
- **All model training runs must be logged to MLflow** — params, metrics, and the model artifact. A training script that doesn't call `mlflow.start_run()` is incomplete.
- **Every prediction must be persisted with a `target_date`** and must be resolvable by the MLOps daily job. Never generate a prediction that isn't written to the `predictions` table.
- **No hardcoded secrets.** All credentials/API keys are read from environment variables via `config/settings.py`, never inlined in source.
- **No mock/placeholder data in the Streamlit dashboard.** Every dashboard view must query live database state.
- **Write unit tests alongside new logic**, especially for: scoring functions (Beneish, Altman), similarity/diff logic, and MARL reward functions. Do not defer tests to "later."
- **Keep local resource usage bounded when services run concurrently.** `embedding_service.py` must embed in small batches (16–32 chunks) so CPU/RAM doesn't spike when processing a full filing. `optuna_search_space.py`/`lightgbm_trainer.py` must run Optuna with `n_jobs=1` (or an explicit, documented trial cap) rather than defaulting to all-core parallelism — this machine is also running MySQL, MongoDB, and ChromaDB in Docker simultaneously.

---

## 4. Scope Discipline

- **Do NOT generate frontend UI design work.** Streamlit views should be functional and default-styled only. Custom CSS, theming, component libraries, or polished UX layout is explicitly **out of scope** — it is reserved for a future cloned template that will replace the dashboard layer entirely. If asked to "make the dashboard look nicer," push back and point to this rule.
- **Do NOT add real-money trading, brokerage integration, or order execution code**, under any framing (not even "for testing," not even behind a feature flag). This system produces research signals and simulations only.
- **Do NOT expand data sources beyond yfinance and SEC EDGAR** without an explicit instruction from a human maintainer — no scraping other sites, no third-party financial data APIs (even free ones), unless this document is updated first.
- **Do NOT add authentication, multi-tenancy, or billing.** This is a single-operator local deployment.
- Stay within the **phase boundaries defined in `phases.md`**. Do not build Phase 5 (MARL) logic while working a Phase 3 (Quant) ticket, even if it seems convenient — cross-phase shortcuts break the testability guarantees in `phases.md`.
- **Phase 5's MARL exit bar is numerical stability, not policy convergence quality.** The agents are not required to reach a learned equilibrium — only to produce bounded, non-NaN drawdown/profit output. Do not spend effort tuning agents toward "realistic" learned behavior; that is out of scope for v1. Use discrete action spaces (not continuous) and apply `np.nan_to_num`-style boundary checks in `reward_functions.py` as a hard safeguard against degenerate/NaN outputs.

---

## 5. Data Integrity Rules

- SEC EDGAR calls **must** include a compliant `User-Agent` header (`AQQIP research-bot <contact-email>`) — EDGAR will block non-compliant requests.
- Never delete raw ingested data (`sec_filings` raw text, `ohlcv_prices` rows) as part of a "cleanup" — raw data is the audit trail. Derived/computed tables (`fundamental_ratios`, `semantic_diffs`) may be recomputed and overwritten, but raw ingestion tables are append-only/upsert-only.
- Any schema change to MySQL tables must be delivered as a new numbered migration file in `db/mysql/init/`, never as an in-place edit to an existing migration that has already been applied.
- ChromaDB collections are named `filings_{ticker}` — do not consolidate into a single global collection; chronological filtering depends on clean per-ticker metadata.
- **Filing text must be HTML-stripped before section extraction.** 10-K/10-Q source is frequently raw HTML or inline XBRL, not plain text. `nltk_preprocessor.py` must run `raw_text` through `BeautifulSoup4` (or `html2text`) before any regex/heading-based "Item 1A" extraction — never regex against unstripped markup.
- **Section extraction must have a length-based fallback check.** If the extracted "Risk Factors" section is under 500 words, treat primary extraction as failed and trigger a secondary parsing strategy rather than persisting a truncated section silently.
- **Failed extractions must be recorded, not silently skipped.** If extraction (including the fallback) cannot produce a usable section, set the filing's `processing_status` to `extraction_failed` rather than advancing it to `preprocessed`. Never let a filing with failed extraction pass silently into embedding/diffing as if it succeeded.

---

## 6. Disclaimers & Compliance

- Every dashboard view and any generated report/output **must** carry a visible disclaimer that AQQIP is a research/simulation tool and does not constitute investment advice.
- Do not phrase any model output as a direct "buy" or "sell" recommendation. Outputs are framed as probabilities, scores, and simulated scenarios only.

---

## 7. When In Doubt

- If a requested change conflicts with any rule above, **do not silently comply** — flag the conflict to the human maintainer and propose an alternative that stays within constraints.
- If a required capability genuinely cannot be built at $0 cost, document the limitation clearly rather than quietly introducing a paid dependency.
- Prefer boring, well-tested, standard-library-adjacent solutions over clever ones — this is a forensic/audit system; reproducibility and auditability outrank elegance.
