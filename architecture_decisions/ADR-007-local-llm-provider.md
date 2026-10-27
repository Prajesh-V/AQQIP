# ADR-007: Local LLM Provider Abstraction

## Status
Accepted

## Context
AQQIP uses an LLM to interpret semantic-diff evidence (not to generate evidence). The project supports both a local Ollama-hosted model for development/testing and Gemini Free Tier for the external interpretation path.

## Decision
Implement a provider abstraction:

```
LLMProvider (abstract)
    ├── OllamaProvider  → communicates via HTTP to Ollama server
    └── GeminiProvider   → uses Google Gemini Free Tier via LangChain
```

**Primary local model:** Qwen3 8B (Q4_K_M quantization, ~5.2 GB)
**Fallback local model:** Phi-4-mini (~2.5 GB)
**External provider:** Google Gemini via LangChain

The provider used is recorded in every qualitative-diff record (`llm_provider`, `llm_model` fields).

## Alternatives Considered
- Direct Gemini-only → rejected (no offline development capability)
- Ollama Python package as direct dependency → rejected (HTTP API communication is sufficient; avoids unnecessary dependency coupling)
- HuggingFace Transformers for local inference → rejected (Ollama provides simpler model management for the target hardware)

## Rationale
The abstraction prevents model-specific logic from contaminating the semantic-diff pipeline. Local development can iterate on prompts without consuming Gemini free-tier quota. Both providers sit behind the same interface, making comparison benchmarks straightforward.

## Consequences
- Ollama must be installed separately on the host machine (not in Docker) for local LLM testing
- The system must work without any LLM for deterministic detection (LLM is interpretation-only)
- Test suite must not require external Gemini API calls to pass
- Each provider's output must be benchmarked for groundedness before adoption
