# System Variants — Shared Output Contract

All six system variants (S1–S6) take the same input and produce the same output schema. This ensures fair comparison across RAG tiers and agent platforms.

## Input

A single health claim string:

```python
claim: str  # e.g. "Intermittent fasting reverses Type 2 diabetes"
```

## Output Schema

Every variant must return a `FactCheckResult` with these fields:

```python
from pydantic import BaseModel

class FactCheckResult(BaseModel):
    """Shared output schema for all system variants."""

    # --- Required fields (all variants) ---
    claim: str                          # The original input claim
    verdict: str                        # One of the 9 verdicts below
    confidence: float                   # 0.0–1.0
    explanation: str                    # Plain-language explanation of the verdict

    # --- Evidence (empty list for S1) ---
    evidence: list[Evidence]            # Retrieved evidence with sources

    # --- Sub-claims (S3+ decompose; S1/S2 return a single sub-claim = the original claim) ---
    sub_claims: list[SubClaim]          # Decomposed sub-claims with per-claim verdicts

    # --- Safety (all variants) ---
    safety_flags: list[str]             # e.g. ["stopping_medication", "vulnerable_population"]
    is_dangerous: bool                  # True if any safety flag triggered

    # --- Metadata ---
    system: str                         # "S1", "S2", "S3", "S4", "S5", or "S6"
    total_cost_usd: float               # LLM + API costs
    total_duration_seconds: float       # Wall-clock time
```

### Evidence

```python
class Evidence(BaseModel):
    id: str
    source: str                         # "pubmed", "cochrane", "guideline", "drugbank", etc.
    retrieval_method: str               # "api", "cross_encoder", "deep_search", "vlm", "vector_search"
    title: str
    content: str                        # Abstract, passage, or extracted figure data
    url: str | None = None
    study_type: str | None = None       # "rct", "systematic_review", "cohort", etc.
    quality_score: float = 0.0          # 0.0–1.0, from evidence hierarchy
    pmid: str | None = None
```

### SubClaim

```python
class SubClaim(BaseModel):
    id: str
    text: str
    verdict: str | None = None          # Per sub-claim verdict
    evidence: list[str] = []            # Evidence IDs supporting this sub-claim
    confidence: float = 0.0
```

## Verdict Taxonomy

All variants must use this exact set of 9 verdicts:

| Verdict | When to use |
|---------|-------------|
| `SUPPORTED` | Strong evidence from high-quality studies |
| `SUPPORTED_WITH_CAVEATS` | True but requires important context |
| `OVERSTATED` | Kernel of truth, but exaggerated |
| `MISLEADING` | Technically true but gives wrong impression |
| `PRELIMINARY` | Some evidence, too early to confirm |
| `OUTDATED` | Was true, evidence has changed |
| `NOT_SUPPORTED` | No credible evidence found |
| `REFUTED` | Directly contradicted by strong evidence |
| `DANGEROUS` | Could cause harm if followed |

The `Verdict` enum and benchmark mapping utilities live in `src/evaluation/verdict_mapping.py`.

## Per-Variant Expectations

| Variant | Sub-claims | Evidence | Safety | Notes |
|---------|-----------|----------|--------|-------|
| **S1** | Single sub-claim = original claim | Empty list | Required | No retrieval — LLM only |
| **S2** | Single sub-claim = original claim | From vector search | Required | Static corpus, no PICO |
| **S3** | PICO-decomposed | From API + reranker | Required | Function pipeline, no agents |
| **S4** | PICO-decomposed | From API + reranker | Required | LangGraph agents |
| **S5** | PICO-decomposed | From API + reranker | Required | Alt platform agents |
| **S6** | PICO-decomposed | From API + reranker + deep search + VLM | Required | Full agentic RAG |

## Directory Layout

```
systems/
├── __init__.py
├── README.md                  # This file
├── s1_no_retrieval/           # S1: LLM-only baseline
│   ├── __init__.py
│   └── system.py              # verify_claim() stub
├── s2_simple_rag/             # S2: Static corpus + vector search
│   ├── __init__.py
│   └── system.py              # verify_claim() stub
├── s3_advanced_rag/           # S3: Multi-source API + cross-encoder (function pipeline)
│   ├── __init__.py
│   └── system.py              # verify_claim() stub
├── s4_langgraph/              # S4 + S6: LangGraph multi-agent (config-driven)
│   ├── __init__.py
│   ├── agents/                # ReAct agents (retrieval_planner, evidence_retriever, etc.)
│   ├── workflow.py            # LangGraph state graph
│   └── system.py              # verify_claim() entry point
└── s5_alt_platform/           # S5: Alternative agent framework
    ├── __init__.py
    └── system.py              # verify_claim() stub
```

S4 and S6 share `s4_langgraph/` — a config flag enables/disables VLM and deep search.

## Usage

Each variant must expose a `verify_claim` function:

```python
def verify_claim(claim: str) -> FactCheckResult:
    ...
```

This allows the evaluation framework (`src/evaluation/`) to run any variant interchangeably.
