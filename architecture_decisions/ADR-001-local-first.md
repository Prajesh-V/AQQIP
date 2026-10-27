# ADR-001: Local-First Architecture

## Status
Accepted

## Context
AQQIP is designed as a financial-forensics research platform. The project has a hard cost constraint of $0 paid infrastructure. The development machine is an RTX 4060 (8 GB VRAM) / Ryzen 7 7840HS / 16 GB RAM laptop.

## Decision
All infrastructure runs locally in Docker containers or as embedded/local libraries. The only permitted external network calls are to:
- yfinance (market data)
- SEC EDGAR REST API (filing data)
- Google Gemini Free Tier API (optional LLM interpretation)

No AWS, Azure, GCP, Kubernetes, Snowflake, Pinecone paid tier, OpenAI API, or Anthropic API may be introduced.

## Alternatives Considered
- Cloud-hosted databases → rejected (violates $0 constraint)
- Managed ML platforms → rejected (violates $0 constraint)
- SaaS LLM APIs → rejected except Gemini Free Tier

## Rationale
The project is a research/simulation tool for a single operator. Local deployment provides full control, zero cost, and complete data sovereignty. The target machine is powerful enough for all required workloads at the project's intended scale.

## Consequences
- All databases must run in Docker containers
- MLflow uses local SQLite backend
- ChromaDB runs in embedded mode
- GPU-intensive workloads must fit within 8 GB VRAM
- Concurrent service load must be managed carefully
- No horizontal scaling; single-machine only
