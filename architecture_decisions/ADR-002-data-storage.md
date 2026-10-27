# ADR-002: Data Storage Architecture

## Status
Accepted

## Context
AQQIP processes three categories of data: structured time-series (prices, ratios, predictions), unstructured documents (SEC filings, semantic diffs), and vector embeddings (filing chunks). Each category has different query patterns and storage requirements.

## Decision
Use a three-database architecture:
- **MySQL 8.0** for structured/time-series data: OHLCV prices, fundamental ratios, predictions, MARL simulation runs
- **MongoDB 7.0** for document-oriented data: raw SEC filings, filing metadata, semantic diffs, processing state
- **ChromaDB (embedded)** for vector embeddings with per-ticker collections and chronological metadata filtering

MLflow uses its local SQLite backend (`./mlruns/`) for experiment tracking.

## Alternatives Considered
- Single PostgreSQL → rejected (poor fit for unstructured filing documents and vector embeddings)
- SQLite everywhere → rejected (poor concurrency for multiple services)
- Weaviate/Pinecone → rejected (either paid or unnecessary complexity; ChromaDB embedded is sufficient)
- ChromaDB server mode → rejected (unnecessary; embedded mode with shared volume is simpler)

## Rationale
Each database is selected for its strength: MySQL for reliable structured queries with unique constraints; MongoDB for flexible document storage of heterogeneous filing content; ChromaDB for purpose-built vector similarity search. All three run locally in Docker at $0 cost.

## Consequences
- Three databases to manage in Docker Compose
- Data contracts must clearly define which database owns which data
- Cross-database joins are not possible; application-layer coordination required
- All persistence via named Docker volumes
