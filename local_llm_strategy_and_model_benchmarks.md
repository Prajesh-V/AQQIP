# AQQIP Local LLM Strategy — Ollama Testbed and Hardware Plan

## 1. Goal

AQQIP should support a local LLM path for development, offline testing, prompt iteration, and failure testing before the Gemini-backed interpretation path is used for the final cloud-free/free-tier workflow.

The local LLM should not replace the deterministic semantic-diff detector. It is used primarily for the interpretation/summarization layer.

## 2. Target development machine

Assumed developer hardware:

```text
GPU: NVIDIA RTX 4060 Laptop GPU
VRAM: 8 GB
TGP: approximately 140 W
RAM: 16 GB
CPU: AMD Ryzen 7 7840HS-class system as specified by project owner
```

Actual VRAM availability should be checked at runtime because Windows/driver/application overhead consumes part of the nominal 8 GB.

## 3. Primary local model recommendation

### `qwen3:8b`

Use this as the **primary Ollama development model**.

Ollama currently lists `qwen3:8b` as an 8.19B-parameter Q4_K_M model with an approximately 5.2 GB download size and 40K context window. The model supports tool use and a thinking mode. citeturn503172search0turn503172search3

Why it fits AQQIP:

- small enough to leave VRAM headroom;
- stronger reasoning than very small local models;
- suitable for structured/agentic workflows;
- useful for grounded filing-summary tasks;
- Apache 2.0 license on the listed checkpoint;
- can run locally through Ollama.

Recommended initial command:

```bash
ollama pull qwen3:8b
ollama run qwen3:8b
```

## 4. Fallback model

### `phi4-mini`

Ollama lists Phi-4-mini at 3.8B parameters and approximately 2.5 GB for the standard model, with a 128K context window and emphasis on reasoning, mathematics, multilingual use, function calling and resource-constrained environments. citeturn357167search1

Use it when:

- the laptop is under heavy CPU/GPU load;
- Qwen3 context requirements become too expensive;
- a fast local test is more important than maximum local reasoning quality.

Command:

```bash
ollama pull phi4-mini
```

## 5. Models NOT recommended as the default on this machine

### Gemma 3 12B

Ollama lists the Q4_K_M 12.2B model at approximately 8.1 GB. citeturn357167search6

That is too close to an 8 GB VRAM ceiling for a comfortable primary configuration once runtime overhead and context are considered. It may work with partial CPU offload, but that reduces the point of maximizing GPU compute on this machine.

### Mistral Small 3.1 24B

Ollama lists the Q4_K_M version at approximately 15 GB. citeturn357167search4

It is not a sensible default for an 8 GB laptop GPU + 16 GB RAM system.

## 6. Local-versus-Gemini architecture

AQQIP should define an abstraction such as:

```text
LLMProvider
   ├── GeminiProvider
   └── OllamaProvider
```

The rest of the qualitative engine should receive a common interface.

Example:

```text
semantic_diff_agent
        ↓
InterpretationProvider
        ↓
 ┌───────────────┐
 │ Gemini         │
 │ Ollama / Qwen3 │
 └───────────────┘
```

This prevents model-specific logic from contaminating the semantic-diff pipeline.

## 7. Local model role

The local model is allowed to:

- summarize detected changes;
- classify interpretation categories;
- generate analyst-readable explanations;
- test prompt templates;
- test structured JSON output;
- test failure behavior;
- run offline developer experiments.

It must NOT be the authoritative source for:

- filing existence,
- exact sentence additions/deletions,
- cosine similarity,
- Beneish/Altman scores,
- market prices,
- prediction ground truth.

## 8. Prompt contract

Provide the local model only the evidence needed to interpret the change.

Recommended input:

```json
{
  "ticker": "...
  ",
  "filing_year_from": 2024,
  "filing_year_to": 2025,
  "cosine_similarity": 0.71,
  "added_sentences": [],
  "deleted_sentences": [],
  "shifted_sentences": []
}
```

Require structured output such as:

```json
{
  "summary": "...",
  "change_categories": [],
  "evidence_references": [],
  "uncertainties": []
}
```

The model should be instructed to say that evidence is insufficient rather than inventing a reason.

## 9. Local LLM benchmark

Before adopting the local model for development, benchmark:

| Test | Qwen3 8B | Phi-4 Mini | Gemini Free Tier |
|---|---:|---:|---:|
| Evidence-grounded summary | measured | measured | measured |
| JSON validity | measured | measured | measured |
| Unsupported-claim rate | measured | measured | measured |
| Latency | measured | measured | measured |
| VRAM usage | measured | measured | measured |
| Peak RAM | measured | measured | measured |
| Failure under long input | measured | measured | measured |

Do not assume the local model is equivalent to Gemini. The benchmark exists to determine where each provider is appropriate.

## 10. Context-window policy

Even if a model supports a large context window, AQQIP should not send an entire 10-K to the LLM.

The deterministic pipeline should first extract and diff the relevant evidence.

Then the LLM receives:

```text
only relevant changed passages
+
small amount of contextual surrounding text
+
structured metadata
```

This improves speed, reduces hallucination opportunities, and conserves VRAM/RAM.

## 11. GPU optimization policy

For an 8 GB GPU:

- use quantized models;
- keep one LLM loaded at a time during local benchmarking;
- begin with moderate context sizes;
- avoid concurrent Ollama + heavy embedding/model workloads on the same GPU;
- keep Sentence Transformers CPU-only unless a controlled experiment demonstrates a meaningful benefit;
- monitor VRAM/RAM during every benchmark.

The goal is maximum useful compute, not maximum parameter count.

## 12. Recommended model roles

### Primary local interpretation model

```text
Qwen3 8B Q4_K_M
```

### Lightweight fallback

```text
Phi-4-mini
```

### Final cloud/free-tier interpretation path

```text
Google Gemini via LangChain
```

The provider/model name is persisted in every qualitative-diff record.

## 13. Local test modes

### Mode A — deterministic only

No LLM. Verify semantic diff outputs.

### Mode B — Ollama

Run interpretation entirely locally.

### Mode C — Gemini

Run final/free-tier interpretation.

### Mode D — provider comparison

Same evidence bundle → Qwen3 vs Phi-4-mini vs Gemini.

Compare groundedness and latency rather than simply preferring the longest answer.

## 14. Reproducibility

Record:

```text
provider
model
quantization/tag
context length
temperature
seed if supported
prompt template version
input evidence hash
output hash
runtime
VRAM usage
RAM usage
```

## 15. Important implementation decision

Do not make Ollama a mandatory production dependency for the full local deployment unless the final team confirms it is acceptable within the project's infrastructure specification.

The cleanest implementation is:

```text
Production/default interpretation:
Gemini Free Tier

Developer/offline interpretation:
Ollama → Qwen3 8B
```

Both sit behind the same provider interface.

## 16. Why this is the right fit for the 4060

The objective is not to cram the largest possible model into 8 GB of VRAM. A model around 8B parameters at Q4 quantization provides a more balanced point between:

- GPU residency,
- reasoning quality,
- context capacity,
- latency,
- RAM headroom,
- ability to run alongside the rest of the AQQIP stack.

Qwen3 8B is therefore the recommended **maximum practical default**, while Phi-4-mini is the safe low-memory fallback.

## 17. Official model references

- Ollama Qwen3 8B: https://ollama.com/library/qwen3:8b
- Ollama Phi-4-mini: https://ollama.com/library/phi4-mini
- Ollama Gemma 3 12B: https://ollama.com/library/gemma3:12b
- Ollama Mistral Small 3.1: https://ollama.com/library/mistral-small3.1

