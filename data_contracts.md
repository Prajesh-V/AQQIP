# AQQIP Data Contracts & Signal Interfaces

## Purpose

This document defines the canonical data contracts shared by every AQQIP subsystem. The objective is to prevent integration drift: every service must know exactly what it receives, what it produces, how provenance is preserved, and how missing or stale data is handled.

The existing architecture defines MySQL as the structured store, MongoDB as the document store, ChromaDB as the local vector store, and MLflow as the experiment tracker. This document adds the contract layer between those components.

## 1. Core design principles

1. **Raw evidence is immutable in meaning.** Raw SEC filing text and ingested market observations remain the audit trail.
2. **Derived signals are versioned.** Every model-derived output must carry a model/run identifier and generation timestamp.
3. **No silent missingness.** Missing inputs are represented explicitly with status/flags rather than substituted invisibly.
4. **MARL receives the complete evidence bundle.** The simulator should not receive only one opaque score; it receives the separate quantitative and qualitative signals plus provenance and market-state inputs.
5. **Dates are first-class fields.** A prediction cannot exist without a prediction date and target date.
6. **Freshness is explicit.** A user must be able to distinguish newly fetched data from cached data.

## 2. Security identity and company resolution

AQQIP is v1-limited to US SEC filers. A ticker is therefore not the primary identity. The canonical identity is:

```yaml
SecurityIdentity:
  ticker: string
  company_name: string
  cik: string              # 10-digit padded SEC CIK
  exchange: string | null
  security_type: "common_stock"
```

The system must resolve the user's selected ticker to a CIK before filing retrieval. If no valid SEC filer mapping exists, the system must stop the filing-forensics portion and report that the security is unsupported in v1 rather than fabricating a filing relationship.

## 3. Market data contract

```yaml
PriceRecord:
  security:
    ticker: string
    cik: string | null
  trade_date: date
  open: float
  high: float
  low: float
  close: float
  adj_close: float
  volume: int
  source: "yfinance"
  fetched_at: datetime
  freshness_status: "fresh" | "cached" | "stale" | "failed"
```

Rules:

- MySQL natural key: `(ticker, trade_date)`.
- Duplicate ingestion must resolve through upsert behavior.
- No intraday/live-price claim is permitted; v1 uses daily OHLCV.
- The latest available market observation is distinct from the latest filing event.

## 4. SEC filing contract

```yaml
FilingRecord:
  security:
    ticker: string
    cik: string
  accession_number: string
  filing_type: "10-K" | "10-Q"
  filing_date: date
  report_period_end: date | null
  filing_url: string
  primary_document: string
  raw_text: string
  source: "SEC_EDGAR"
  ingested_at: datetime
  processing_status:
    "raw"
    | "preprocessed"
    | "embedded"
    | "diffed"
    | "extraction_failed"
    | "processing_failed"
```

The accession number is the canonical filing identity. Re-ingesting the same filing must not create a duplicate document.

## 5. Financial-ratio contract

```yaml
FundamentalSignal:
  ticker: string
  cik: string
  fiscal_period: string
  beneish_m_score: float | null
  altman_z_score: float | null
  raw_features: object
  computed_at: datetime
  source_filing_accession: string | null
```

Any formula component that cannot be computed must remain `null` and set an explicit feature-completeness flag. Silent zero-filling is prohibited because zero is a legitimate numeric value.

## 6. Quantitative model contract

The quantitative model must expose both its model output and enough metadata to reproduce it.

```yaml
QuantSignal:
  ticker: string
  prediction_date: date
  target_date: date
  model_run_id: string
  model_version: string
  raw_probability: float       # [0,1]
  calibrated_probability: float | null
  quant_pressure_score: float  # [0,1], canonical value used by MARL
  beneish_m_score: float | null
  altman_z_score: float | null
  input_feature_hash: string
  generated_at: datetime
```

`quant_pressure_score` is not a fraud probability. It is a normalized research signal. When calibrated probability is available, the MARL layer should consume the calibrated value rather than treating an uncalibrated classifier output as a probability.

## 7. Qualitative signal contract

```yaml
QualitativeSignal:
  ticker: string
  cik: string
  section: "risk_factors"
  year_from: int
  year_to: int
  accession_from: string
  accession_to: string
  cosine_similarity: float
  semantic_shift_score: float   # [0,1]
  added_sentences: list[string]
  deleted_sentences: list[string]
  shifted_sentences: list[string]
  llm_summary: string
  llm_provider: "gemini" | "ollama"
  llm_model: string
  generated_at: datetime
```

The language model is an interpretation layer. The underlying added/deleted/shifted text and cosine similarity remain authoritative evidence; the LLM narrative is not itself evidence.

## 8. Canonical SignalBundle sent to MARL

This is the most important integration contract in the project.

```yaml
SignalBundle:
  security:
    ticker: string
    company_name: string
    cik: string

  market_context:
    current_price: float
    recent_return: float
    recent_volatility: float
    average_volume: float
    liquidity_proxy: float
    as_of_date: date

  quantitative:
    quant_pressure_score: float
    beneish_m_score: float | null
    altman_z_score: float | null
    model_probability: float | null
    model_run_id: string

  qualitative:
    semantic_shift_score: float
    cosine_similarity: float
    added_count: int
    deleted_count: int
    shifted_count: int
    filing_year_from: int | null
    filing_year_to: int | null
    source_accession: string | null

  evidence_quality:
    quant_available: bool
    qualitative_available: bool
    filing_pair_available: bool
    feature_completeness: float
    data_freshness_minutes: float | null

  fusion:
    evidence_concordance: float
    combined_forensic_pressure: float
    fusion_version: string

  provenance:
    prediction_id: int | null
    quant_model_run_id: string | null
    qualitative_diff_id: string | null
    random_seed: int
```

### Why MARL receives separate signals

AQQIP should not collapse every inference into one irreversible number before simulation. The simulator needs to know whether the evidence is:

- strongly quantitative but weakly qualitative,
- weakly quantitative but strongly qualitative,
- strong in both channels, or
- missing in one channel.

That enables the ablation experiments and lets the dashboard explain *why* a scenario changed.

## 9. Signal fusion rule

The v1 fusion layer should produce a derived `combined_forensic_pressure` but preserve the raw signals.

Recommended initial form:

```text
combined_forensic_pressure = w_q * quant_pressure_score
                           + w_l * semantic_shift_score
```

with `w_q + w_l = 1`.

Start with equal weights for the initial controlled experiment unless an empirical calibration study supports another choice. Do not silently change weights between experiments.

Also compute:

```text
evidence_concordance = 1 - abs(Q - L)
```

This is a measure of agreement between the two signals, not an independent probability.

## 10. MARL output contract

```yaml
MARLResult:
  simulation_id: string
  ticker: string
  scenario_seed: int
  episode_days: int
  initial_price: float
  final_price: float
  max_drawdown_pct: float
  price_volatility: float
  short_seller_pnl_pct: float
  retail_herd_pnl_pct: float
  market_maker_pnl_pct: float
  market_maker_inventory_peak: float
  terminated_reason: string
  action_trace_uri_or_reference: string | null
  scenario_params: object
  signal_bundle_hash: string
  model_version: string
  run_at: datetime
```

## 11. Prediction contract

```yaml
PredictionRecord:
  prediction_id: int
  ticker: string
  model_run_id: string
  predicted_drop_probability: float
  predicted_at: datetime
  target_date: date
  actual_drop_observed: bool | null
  actual_price_change_pct: float | null
  resolved_at: datetime | null
  status: "open" | "resolved"
```

The precise prediction horizon/threshold must be frozen in `evaluation_protocol.md` before final benchmarking. The existing architecture only requires a target-date-based outcome resolver.

## 12. Missing-data behavior

| Condition | Required behavior |
|---|---|
| No recent price | Do not run MARL; show data unavailable |
| No previous 10-K | Qualitative score = unavailable; allow quantitative-only analysis |
| Failed Risk Factors extraction | Mark extraction failure; never embed truncated text |
| Quant model unavailable | Allow qualitative-only review but MARL must record missing quantitative evidence |
| Qualitative model unavailable | Allow quantitative-only review but MARL must record missing qualitative evidence |
| Both unavailable | Analysis request fails safely |
| Stale price data | Show freshness warning and require user confirmation for scenario run |

## 13. Contract versioning

Every cross-service payload should carry a `contract_version` or equivalent schema version. A schema change is a breaking change unless the consumer can safely ignore the new field.

