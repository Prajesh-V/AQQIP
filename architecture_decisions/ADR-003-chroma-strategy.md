# ADR-003: ChromaDB Collection Strategy

## Status
Accepted

## Context
AQQIP embeds SEC filing text (Risk Factors sections) for chronological semantic comparison. The system must efficiently retrieve all chunks from a specific ticker and fiscal year to compute Year N vs Year N-1 diffs.

## Decision
Use per-ticker ChromaDB collections named `filings_{ticker}`. Chronological separation is achieved via metadata filtering (`fiscal_year`, `filing_type`) rather than separate physical collections per year.

Each embedded chunk carries metadata:
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

## Alternatives Considered
- Single global collection → rejected (metadata filtering becomes expensive at scale; per-ticker separation is cleaner)
- Separate collection per year → rejected (doubles embedding pipeline complexity; metadata filtering is sufficient)
- External vector DB service → rejected (violates $0 constraint)

## Rationale
Per-ticker collections keep chronological filtering cheap and avoid cross-ticker interference. Metadata filtering on `fiscal_year` within a collection provides the needed Year N vs N-1 retrieval without duplicate embedding pipelines.

## Consequences
- Collection naming convention must be respected across ingestion and query code
- New tickers create new collections automatically
- Embedding model must remain consistent within a collection (all-MiniLM-L6-v2)
- Historical embeddings are not rebuilt when a new filing arrives
