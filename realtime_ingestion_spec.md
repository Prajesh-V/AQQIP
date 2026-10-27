# AQQIP Near-Real-Time SEC & Market Data Ingestion Specification

## 1. Goal

AQQIP should behave like a continuously maintained research system rather than a static dataset importer.

The correct claim for v1 is **near-real-time filing awareness**, not a live exchange-grade trading feed.

The SEC states that its EDGAR submissions history and XBRL APIs are updated throughout the day in real time, with typical processing delays under a second for submissions and under a minute for XBRL, although delays can increase during peak periods. SEC RSS feeds are also provided for staying current with new EDGAR filings. citeturn230039search0turn230039search2

AQQIP therefore uses polling/event-detection for SEC filings plus daily OHLCV ingestion from yfinance.

## 2. Two different freshness domains

### Filing freshness

10-K/10-Q detection can be near-real-time because EDGAR exposes submission history continuously.

### Market-price freshness

v1 uses daily OHLCV from yfinance. This is **not** an intraday real-time market feed.

The dashboard must therefore show two timestamps:

```text
Latest filing observed: YYYY-MM-DD HH:MM
Latest market bar:      YYYY-MM-DD
```

Never imply that the stock price is live.

## 3. User-configured watchlist

The tracked universe is stored in configuration/database state.

Example:

```yaml
watchlist:
  - ticker: AAPL
    enabled: true
  - ticker: MSFT
    enabled: true
```

For each ticker, resolve and store:

```text
Ticker
Company name
CIK
Exchange
Last submissions check
Last filing accession seen
Last price date seen
```

## 4. Filing discovery loop

### Preferred source

Use SEC `data.sec.gov/submissions/CIK##########.json` for the tracked CIKs.

The SEC documents that each entity's submissions history contains recent filing metadata, including form and filing dates, and that the JSON structures are updated in real time. citeturn230039search0

### Secondary discovery

SEC EDGAR RSS can be used as a supplementary signal for newly published filings, especially for testing near-real-time notification behavior. citeturn230039search2

The submission history remains the canonical source used to establish whether an accession is new.

## 5. Recommended polling behavior

For a small tracked universe, use a configurable polling interval, for example:

```text
normal mode: 60–120 seconds
backoff mode: 5–15 minutes after repeated failures
manual refresh: user-triggered for one ticker
```

These are project configuration recommendations, not SEC requirements.

SEC fair-access guidance currently limits users to no more than 10 requests/second and requires a declared User-Agent for scripted access. AQQIP should operate orders of magnitude below that limit and use exponential backoff on 429/503 responses. citeturn368051search0turn368051search1turn368051search6

## 6. Filing detection algorithm

For each tracked ticker:

```text
1. Resolve CIK.
2. Request current submissions JSON.
3. Filter forms to 10-K and 10-Q.
4. Sort by filing date/accession.
5. Compare accession numbers to known MongoDB records.
6. For unseen accession:
      a. create a discovery record
      b. obtain filing metadata/document location
      c. download primary filing document
      d. persist raw text + metadata
      e. enqueue preprocessing
7. Update last-seen accession/check timestamp.
```

The accession number is the idempotency key.

## 7. Never re-download unchanged filings

A filing already present under the accession number should not be processed again unless:

- processing previously failed,
- a human explicitly requests reprocessing,
- or the parser/version changed and a migration/reprocessing job was launched.

This avoids wasting SEC requests and model compute.

## 8. Filing processing pipeline

```text
NEW ACCESSION
     ↓
RAW FILING FETCH
     ↓
MONGODB sec_filings
processing_status = raw
     ↓
HTML/XBRL CLEANING
     ↓
SECTION EXTRACTION
     ↓
Risk Factors
MD&A where supported
Notes where supported
     ↓
VALIDATION
     ↓
CHUNKING
     ↓
EMBEDDING
     ↓
CHROMADB
     ↓
YEAR-N vs YEAR-(N-1) DIFF
     ↓
SEMANTIC SHIFT SCORE
     ↓
LOCAL/REMOTE LLM INTERPRETATION
     ↓
MONGODB semantic_diffs
```

The existing AQQIP rules require HTML stripping before section extraction, a length-based fallback when Risk Factors extraction looks suspiciously short, and explicit `extraction_failed` state instead of silently passing bad text downstream.

## 9. Extraction quality gate

Before embedding, validate:

- filing type is 10-K or 10-Q;
- text is non-empty;
- Risk Factors heading was found using primary or fallback extraction;
- extracted text exceeds the configured minimum quality threshold;
- extracted section is not obviously an HTML/markup artifact;
- accession and filing date are present.

If the gate fails:

```text
processing_status = extraction_failed
```

and the filing must not enter ChromaDB.

## 10. Quarterly versus annual filing handling

10-K and 10-Q must remain distinguishable.

Do not compare:

```text
10-K FY2025 ↔ 10-Q Q1-2026
```

as if they were equivalent documents.

Recommended comparison hierarchy:

```text
10-K year N   ↔ 10-K year N-1
10-Q Qx year N ↔ 10-Q Qx year N-1
```

Fallback comparisons must be explicitly labelled and excluded from benchmarks requiring like-for-like filings.

## 11. Quantitative data refresh

A filing event should trigger a controlled quantitative refresh because the filing may update financial facts.

Sequence:

```text
10-K/10-Q detected
       ↓
Update financial facts
       ↓
Recompute Beneish / Altman features
       ↓
Generate next eligible prediction
       ↓
Store model_run_id + target_date
```

Do not retroactively change a historical prediction's feature vector. Instead, create a new prediction record with the new `prediction_date` and model run.

## 12. User-triggered "Analyze Now"

The dashboard should expose a manual refresh path.

When a user clicks `Analyze Now`:

```text
1. Check last filing seen.
2. Check last price observation.
3. Trigger a targeted refresh if stale.
4. Wait for ingestion completion with a bounded timeout.
5. Run quant + qualitative pipeline.
6. Create SignalBundle.
7. Offer MARL scenario.
```

If the SEC request fails, use the last known valid filing but clearly label the analysis as cached/stale. Do not pretend freshness.

## 13. Background processing architecture

The local system should separate:

```text
DISCOVERY
  ↓
INGESTION QUEUE
  ↓
PROCESSING QUEUE
  ↓
ANALYSIS
  ↓
SIMULATION
```

For v1, this can be implemented without introducing a paid message broker. A database-backed job/status table or lightweight local scheduler is sufficient.

## 14. SEC rate-limit and retry policy

All SEC requests must:

- provide a compliant User-Agent;
- use Tenacity or equivalent exponential backoff;
- treat 429/503 as retryable;
- cap retries;
- persist a failure record;
- avoid tight polling loops.

The SEC explicitly documents 429 as a possible too-frequent request response and expects a User-Agent identifying the client/application. citeturn368051search6

## 15. End-to-end filing event latency measurement

Record these timestamps:

```text
T0 = SEC filing becomes discoverable
T1 = AQQIP detects accession
T2 = raw filing stored
T3 = Risk Factors extracted
T4 = embeddings stored
T5 = semantic diff complete
T6 = quant refresh complete
T7 = analysis-ready
```

Then compute:

```text
Detection latency = T1 - T0
Processing latency = T7 - T2
End-to-end readiness = T7 - T0
```

The system should measure these before claiming real-time/near-real-time performance.

## 16. Failure/recovery model

Each ingestion item has a state machine:

```text
DISCOVERED
   ↓
FETCHING
   ↓
RAW_STORED
   ↓
PREPROCESSING
   ↓
EMBEDDING
   ↓
DIFFING
   ↓
READY
```

Failure states:

```text
FETCH_FAILED
EXTRACTION_FAILED
EMBEDDING_FAILED
DIFF_FAILED
```

A failed item can be retried without duplicating the raw record.

