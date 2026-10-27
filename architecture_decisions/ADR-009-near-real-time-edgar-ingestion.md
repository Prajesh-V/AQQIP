# ADR-009: Near-Real-Time EDGAR Ingestion

## Status
Accepted

## Context
AQQIP must detect new 10-K/10-Q filings for tracked companies without requiring manual uploads. SEC EDGAR provides a submissions API that is updated in real-time, plus RSS feeds for newly published filings.

## Decision
Use **polling-based filing discovery** against the SEC submissions API:
- Primary source: `data.sec.gov/submissions/CIK##########.json`
- Secondary/supplementary: SEC EDGAR RSS feeds
- Polling interval: configurable (default 60–120 seconds for normal mode)
- Backoff mode: 5–15 minutes after repeated failures
- Manual refresh: user-triggered for a single ticker

The accession number is the idempotency key. A filing already stored by accession number is never re-downloaded unless processing previously failed.

## Alternatives Considered
- Webhook/push notification from SEC → not available
- Continuous streaming → not provided by SEC EDGAR; polling is the only option
- RSS-only discovery → rejected as primary (submissions API is more reliable and structured)
- Aggressive polling (<10s) → rejected (violates SEC fair-access guidelines of max 10 req/s and wastes resources)

## Rationale
The SEC explicitly documents that submissions history is updated in real-time. Polling at 60–120 second intervals for a small tracked universe stays well within fair-access limits while providing near-real-time filing awareness. All requests include a compliant User-Agent and use Tenacity-based exponential backoff on 429/503 responses.

## Consequences
- Filing detection latency depends on polling interval (minimum ~60s in normal mode)
- Rate limiting must be respected (max 10 req/s; AQQIP operates far below this)
- Each polling cycle checks only tracked CIKs, not the entire EDGAR universe
- New filing discovery triggers the full processing pipeline: download → extract → embed → diff → score
- The system must not re-download unchanged filings
