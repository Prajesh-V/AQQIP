# AQQIP — Autonomous Quantitative & Qualitative Investment Platform

**A Multi-Agent LLM Framework for Autonomous Financial Forensics and Market Simulation**

> ⚠️ AQQIP is a financial research and simulation tool. Outputs are model-generated
> signals and simulated scenarios, not investment advice or fraud verdicts.

---

## 1. Project Purpose

AQQIP is a local-first financial research platform that:

1. **Ingests** public market data (yfinance) and SEC filings (EDGAR)
2. **Detects** quantitative anomalies via Beneish M-Score, Altman Z-Score, and a LightGBM classifier
3. **Compares** consecutive SEC filings semantically to identify disclosure language changes
4. **Fuses** quantitative and qualitative evidence into a versioned SignalBundle
5. **Simulates** market reactions via a multi-agent reinforcement learning (MARL) environment
6. **Tracks** predictions against real outcomes and retrains models through an MLOps feedback loop
7. **Presents** all evidence and simulations through a Streamlit dashboard

AQQIP is **NOT** a brokerage, live trading engine, investment advisor, or fraud-verdict system.

---

## 2. Architecture Overview

```
User → Streamlit Dashboard (8501)
          ↓
  Ticker/CIK Resolution
          ↓
  Data Freshness Gate
          ↓
  ┌───────────────────────────────┐
  │ Ingestion Service             │
  │  yfinance → MySQL             │
  │  SEC EDGAR → MongoDB          │
  └───────────────────────────────┘
          ↓                   ↓
  ┌──────────────┐   ┌──────────────────┐
  │ Quant Engine │   │ Qualitative Engine│
  │  Beneish     │   │  NLP + Embeddings │
  │  Altman      │   │  Semantic Diff    │
  │  LightGBM    │   │  LLM Interpret    │
  └──────────────┘   └──────────────────┘
          ↓                   ↓
         Signal Fusion (SignalBundle)
                    ↓
          MARL Simulation Engine
                    ↓
          MLOps Feedback Loop
                    ↓
          MLflow Experiment Tracking (5000)
```

### Database Architecture
| Database | Purpose | Port |
|---|---|---|
| MySQL 8.0 | Prices, ratios, predictions, MARL runs | 3306 |
| MongoDB 7.0 | SEC filings, semantic diffs | 27017 |
| ChromaDB | Filing embeddings (embedded mode) | — |
| MLflow | Experiment tracking (SQLite backend) | 5000 |

---

## 3. Hardware Assumptions

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM, ~140W) |
| RAM | 16 GB |
| CPU | AMD Ryzen 7 7840HS |

The project is designed to run within these resource constraints. Embeddings use CPU. Local LLM uses Ollama with quantized models (Qwen3 8B Q4_K_M primary, Phi-4-mini fallback).

---

## 4. Software Prerequisites

- **Docker Desktop** with Docker Compose v2
- **Python 3.11+** (for local development/testing outside Docker)
- **Git**
- **Ollama** (optional, for local LLM testing)

---

## 5. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd AAQIP

# Create environment file from template
cp .env.example .env

# Edit .env with your settings (especially GEMINI_API_KEY if using Gemini)
```

### Required Environment Variables

| Variable | Description | Required |
|---|---|---|
| `MYSQL_ROOT_PASSWORD` | MySQL root password | Yes (has default) |
| `MYSQL_DATABASE` | MySQL database name | Yes (default: aqqip) |
| `MYSQL_USER` / `MYSQL_PASSWORD` | MySQL application credentials | Yes (have defaults) |
| `GEMINI_API_KEY` | Google Gemini Free Tier API key | No (optional) |
| `EDGAR_USER_AGENT` | SEC-compliant User-Agent | Yes (has default) |
| `OLLAMA_HOST` | Ollama server URL | No (optional) |

See [`.env.example`](.env.example) for the complete list.

---

## 6. Docker Compose Startup

```bash
# Start all services
docker compose up -d

# Verify services are healthy
docker compose ps

# Check logs
docker compose logs -f mysql mongodb

# Stop services (data persists in volumes)
docker compose down

# Stop and remove volumes (DESTROYS DATA)
docker compose down -v
```

### Service URLs

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| MySQL | localhost:3306 |
| MongoDB | localhost:27017 |

---

## 7. Health Verification

```bash
# Run infrastructure health checks (requires Docker services running)
python -m pytest tests/test_infra_healthcheck.py -v

# Run all tests
python -m pytest tests/ -v
```

---

## 8. MLflow Access

MLflow UI is available at http://localhost:5000 after `docker compose up`.

MLflow uses a local SQLite backend (`./mlruns/`). No remote tracking server or cloud storage.

---

## 9. Streamlit Dashboard Access

The dashboard is at http://localhost:8501. Currently shows a Phase 1 placeholder.

**Current phase:** Infrastructure only. Business views (ticker search, evidence, MARL simulation) will be implemented in Phase 9.

---

## 10. Ollama Setup for Local LLM Testing

Ollama is **optional** and runs on the host machine (not in Docker):

```bash
# Install Ollama (see https://ollama.com)

# Pull the primary model
ollama pull qwen3:8b

# Pull the fallback model
ollama pull phi4-mini

# Verify
ollama run qwen3:8b "Hello"
```

Set `OLLAMA_HOST=http://localhost:11434` in `.env` (or `http://host.docker.internal:11434` for Docker services to reach the host).

---

## 11. Repository Structure

```
AAQIP/
├── architecture.md              # System architecture
├── data_contracts.md            # Typed signal interfaces
├── evaluation_protocol.md       # Benchmarks and metrics
├── local_llm_strategy_and_model_benchmarks.md
├── marl_design_spec.md          # MARL environment design
├── marl_experiment_matrix.md    # Experiment definitions
├── memory.md                    # AI coding assistant rulebook
├── mlops_contract.md            # MLOps feedback specification
├── phases.md                    # Phase definitions
├── plan.md                      # Sprint checklist
├── prd.md                       # Product requirements
├── realtime_ingestion_spec.md   # EDGAR polling specification
├── user_workflow_and_ticker_selection.md
│
├── architecture_decisions/      # ADRs (001–009)
├── config/
│   └── settings.py              # Centralized typed configuration
├── db/
│   ├── mysql/init/              # MySQL migration scripts
│   └── mongo/init/              # MongoDB initialization
├── src/
│   ├── database/                # Database access layer
│   ├── ingestion/               # yfinance + EDGAR fetchers
│   ├── quant/                   # Quantitative engine
│   ├── qualitative/             # Qualitative engine
│   ├── marl/                    # MARL simulation engine
│   ├── mlops/                   # MLOps feedback loop
│   ├── utils/                   # Shared utilities
│   └── stub_service.py          # Phase 1 placeholder
├── dashboard/
│   └── app.py                   # Streamlit application
├── tests/                       # Test suite
├── logs/                        # Application logs (gitignored)
│
├── docker-compose.yml
├── Dockerfile.python
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 12. Current Implementation Phase

**Phase 1: Infrastructure — COMPLETE**

| Phase | Status | Description |
|---|---|---|
| Phase 0 | ✅ Complete | Repository reconnaissance |
| Phase 1 | ✅ Complete | Infrastructure & databases |
| Phase 2 | 🔲 Planned | Data ingestion pipeline |
| Phase 3 | 🔲 Planned | Quantitative engine |
| Phase 4 | 🔲 Planned | Qualitative engine |
| Phase 5 | 🔲 Planned | Signal fusion |
| Phase 6 | 🔲 Planned | MARL environment |
| Phase 7 | 🔲 Planned | MARL experiments |
| Phase 8 | 🔲 Planned | MLOps feedback loop |
| Phase 9 | 🔲 Planned | Dashboard views |
| Phase 10 | 🔲 Planned | Full system validation |

---

## 13. Project Boundaries

### This system IS:
- A financial research and forensics platform
- A controlled market-reaction simulator
- A self-correcting prediction tracking system
- A local-first, $0-cost deployment

### This system is NOT:
- A brokerage or trading engine
- Investment advice or a buy/sell recommendation system
- A fraud-verdict generator
- A real-time market data feed
- A production financial execution system
- A SaaS application

### Cost Constraint
**$0 paid infrastructure.** The only external network calls are to yfinance, SEC EDGAR, and Google Gemini Free Tier. No AWS, Azure, GCP, Kubernetes, Snowflake, OpenAI API, or Anthropic API.

---

## License

This project is for research purposes.
