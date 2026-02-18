# Health Claims Fact-Checker

## Project Overview

Multi-agent system for verifying health claims against peer-reviewed research, clinical trials, and medical guidelines. Uses **multi-method retrieval** (not just vector DB) — agents intelligently choose the right retrieval method per sub-claim.

**Core insight:** RAG ≠ Vector Database. We use API search, cross-encoder re-ranking, targeted embedding, and VLM — picking the right method for each query.

**Two comparison threads:**
1. **RAG Tiers** — Simple RAG vs Advanced RAG vs Agentic RAG (when does each tier add value?)
2. **Agent Architectures** — Same pipeline on LangGraph vs alternative platform vs function pipeline (does the framework matter?)

System is general-purpose; **evaluation focuses on vaccine misinformation** (clearer ground truth, high-impact domain).

## Quick Start

```bash
# Setup (uses uv for dependency management, pins Python 3.12)
uv sync

# Install scispaCy biomedical model
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Register Jupyter kernel (for notebooks)
uv run python -m ipykernel install --user --name health-claim-checker --display-name "Health Claim Checker (Python 3.12)"

# Download benchmarks
uv run python scripts/download_benchmarks.py

# Index guidelines (one-time)
uv run python scripts/index_guidelines.py

# Run the app
uv run streamlit run app/streamlit_app.py
```

## Architecture

```
Claim → Decomposer → Retrieval Planner → Evidence Retriever → VLM Extractor
                                                    ↓
                              Verdict Agent ← Evidence Grader ← Safety Checker
```

### 7 Nodes (5 Agents + 2 Functions)

| Node | Type | Purpose | Model |
|------|------|---------|-------|
| Claim Decomposer | Function | PICO extraction + sub-claims (fixed steps) | Claude Sonnet |
| Retrieval Planner | Agent (ReAct) | Reasons about which retrieval method per sub-claim | Claude Sonnet |
| Evidence Retriever | Agent (ReAct) | Multi-method search with adaptive strategy | APIs + PubMedBERT |
| VLM Extractor | Agent (ReAct) | Reasons about figure type, extracts data accordingly | Claude Vision |
| Evidence Grader | Agent (ReAct) | Reasons about study quality, bias, conflicts | Claude Sonnet |
| Verdict Agent | Agent (ReAct) | Weighs evidence, reasons about nuance, synthesizes verdict | Claude Sonnet |
| Safety Checker | Function | Flag dangerous claims (pattern matching) | Claude Sonnet |

### Multi-Method Retrieval (NOT just vector DB)

| Stage | Method | Embedding? |
|-------|--------|-----------|
| Discovery | PubMed API, Semantic Scholar, Cochrane, ClinicalTrials.gov | No |
| Ranking | Cross-encoder re-ranker | No |
| Deep Search | PubMedBERT on full-text chunks (~500/claim) | Yes (small, ephemeral) |
| Guidelines | Pre-indexed vector store (~18K chunks) | Yes (persistent) |
| Structured | DrugBank API, direct doc fetch | No |
| Visual | Claude Vision on PDF figures | No |

**Embedding is used in 1 of 5 stages, over small corpora.**

## Comparison Framework

### Thread 1: RAG Tiers

| Tier | System | What It Does | Embedding? |
|------|--------|-------------|-----------|
| **Simple RAG** | Embed vaccine articles → vector search → LLM verdict | Naive chunk-and-retrieve | Yes (static corpus) |
| **Advanced RAG** | PICO query reformulation → Multi-source API → Cross-encoder rerank → LLM verdict | Intelligent retrieval without agents | Minimal (reranker only) |
| **Agentic RAG** | ReAct agent → API discovery → rerank → deep search → VLM → grading → verdict | Agent-driven adaptive retrieval | Yes (ephemeral, targeted) |

### Thread 2: Agent Architectures

Same pipeline implemented on multiple platforms. Team picks 1-2 alternatives alongside LangGraph; function pipeline included as no-agent baseline.

| Platform | Style |
|----------|-------|
| **LangGraph** (current) | Structured state graph, typed state, conditional routing |
| **CrewAI / AutoGen / smolagents / PydanticAI** | Team picks 1-2 alternatives |
| **Function Pipeline** (baseline) | Plain Python functions, no agent reasoning |

### 6 System Variants

| # | System | RAG Tier | Agent Arch | Purpose |
|---|--------|----------|-----------|---------|
| S1 | No retrieval | None | None | LLM knowledge baseline |
| S2 | Simple RAG | Simple | None | Naive RAG baseline |
| S3 | Advanced RAG | Advanced | Function pipeline | Multi-source retrieval without agents |
| S4 | Multi-Agent (LangGraph) | Advanced | LangGraph | Agent orchestration value |
| S5 | Multi-Agent (Alt Platform) | Advanced | Alt framework | Platform comparison |
| S6 | Full Agentic RAG | Agentic | LangGraph | Full system with deep search + VLM |

**Key comparisons:** S1→S2 (retrieval value), S2→S3 (advanced retrieval value), S3→S4 (agent value), S4→S5 (platform comparison), S4→S6 (agentic RAG value).

## Directory Structure

```
health-claim-checker/
├── CLAUDE.md                 # This file
├── README.md
├── pyproject.toml            # Dependencies (managed by uv)
├── .python-version           # Python 3.12 pin
├── requirements.txt          # Legacy reference (use pyproject.toml)
├── .env.example              # API keys template
│
├── data/
│   ├── benchmarks/           # SciFact, PUBHEALTH, HealthVer, ANTi-Vax (downloaded)
│   │   ├── scifact/
│   │   ├── pubhealth/
│   │   ├── healthver/
│   │   └── antivax/
│   ├── claims/
│   │   ├── curated_claims.json       # ~200 vaccine-focused claims
│   │   ├── perturbed_claims.json
│   │   ├── ground_truth.json
│   │   └── pico_ground_truth.json    # 30 hand-labeled claims for PICO eval
│   ├── guidelines/           # Pre-downloaded PDFs
│   │   ├── who/
│   │   ├── nih/
│   │   └── moh_singapore/
│   └── figures/              # Medical figures for VLM eval
│       ├── forest_plots/
│       ├── kaplan_meier/
│       └── ground_truth.json
│
├── src/                          # SHARED LIBRARY — all variants import from here
│   ├── __init__.py
│   ├── config.py                 # Settings, API keys, model configs
│   ├── models.py                 # Shared data models (PICO, SubClaim, Evidence, etc.)
│   │
│   ├── functions/                # Single-pass nodes (no reasoning loop)
│   │   ├── __init__.py
│   │   ├── decomposer.py        # Claim → PICO + sub-claims
│   │   └── safety_checker.py    # Dangerous claim detection
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pubmed_client.py     # PubMed E-utilities wrapper
│   │   ├── semantic_scholar.py  # Semantic Scholar API
│   │   ├── cochrane_client.py   # Cochrane search
│   │   ├── clinical_trials.py   # ClinicalTrials.gov API
│   │   ├── drugbank_client.py   # Drug info API
│   │   ├── cross_encoder.py     # Abstract re-ranking
│   │   ├── deep_search.py       # On-the-fly full-text embedding
│   │   ├── guideline_store.py   # Pre-indexed guideline vector DB
│   │   └── trust_ranker.py      # Evidence hierarchy scoring
│   │
│   ├── medical_nlp/
│   │   ├── __init__.py
│   │   ├── pico_extractor.py    # PICO element extraction (hallucination fix applied)
│   │   ├── medical_ner.py       # Drug, condition, gene NER
│   │   └── mesh_mapper.py       # Map to MeSH vocabulary
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── decomposition_eval.py
│       ├── retrieval_planning_eval.py
│       ├── retrieval_eval.py
│       ├── vlm_eval.py
│       ├── grading_eval.py
│       ├── verdict_eval.py
│       ├── safety_eval.py
│       └── tool_selection_eval.py  # Agent tool selection accuracy
│
├── systems/                      # VARIANT IMPLEMENTATIONS — symmetric, independent
│   ├── __init__.py
│   ├── README.md                 # Shared output contract
│   ├── s1_no_retrieval/          # S1: LLM-only baseline
│   │   ├── __init__.py
│   │   └── system.py
│   ├── s2_simple_rag/            # S2: Static corpus + vector search
│   │   ├── __init__.py
│   │   └── system.py
│   ├── s3_advanced_rag/          # S3: Multi-source API + cross-encoder (function pipeline)
│   │   ├── __init__.py
│   │   └── system.py
│   ├── s4_langgraph/             # S4 + S6: LangGraph multi-agent (config-driven)
│   │   ├── __init__.py
│   │   ├── agents/               # ReAct agents (retrieval_planner, evidence_retriever, etc.)
│   │   ├── workflow.py           # LangGraph state graph
│   │   └── system.py             # verify_claim() entry point
│   └── s5_alt_platform/          # S5: Alternative agent framework
│       ├── __init__.py
│       └── system.py
│
├── app/
│   └── streamlit_app.py      # Multi-tab demo UI
│
├── notebooks/
│   ├── 01_api_exploration.ipynb
│   ├── 02_retrieval_comparison.ipynb  # Retrieval planner + PICO eval (30 claims, 3 metrics)
│   ├── 03_vlm_figure_extraction.ipynb
│   ├── 04_scifact_baseline.ipynb
│   └── 05_results_analysis.ipynb
│
├── scripts/
│   ├── download_benchmarks.py
│   ├── index_guidelines.py
│   ├── extract_figures.py
│   ├── run_baselines.py
│   └── run_evaluation.py
│
└── tests/
    ├── test_agents/
    ├── test_retrieval/
    └── test_evaluation/
```

## Key Dependencies

Managed via `pyproject.toml` + `uv sync`. Requires Python >=3.12, <3.13.

```
# Core
langchain, langgraph, anthropic, openai, pydantic

# Retrieval
sentence-transformers, faiss-cpu, biopython

# Medical NLP
scispacy  # + en_core_sci_sm model installed separately via uv pip

# Data
pandas, pdfplumber, requests

# UI
streamlit

# Observability
langfuse

# Dev
pytest, python-dotenv, ipykernel
```

## API Keys Required

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...      # Claude for agents + VLM
NCBI_API_KEY=...                   # PubMed (optional, higher rate limit)
SEMANTIC_SCHOLAR_API_KEY=...       # Optional, higher rate limit
```

## Evidence Hierarchy (Weights)

```python
EVIDENCE_WEIGHTS = {
    "guideline": 1.0,           # WHO, NIH, MOH
    "systematic_review": 0.9,   # Cochrane, meta-analyses
    "rct": 0.8,                 # Randomized controlled trials
    "cohort": 0.6,              # Cohort studies
    "case_control": 0.5,        # Case-control studies
    "case_report": 0.3,         # Case reports
    "in_vitro": 0.2,            # Lab studies
    "expert_opinion": 0.1,      # Expert opinion
}
```

## Verdict Taxonomy

```python
VERDICTS = [
    "SUPPORTED",              # Strong evidence
    "SUPPORTED_WITH_CAVEATS", # True but needs context
    "OVERSTATED",             # Kernel of truth, exaggerated
    "MISLEADING",             # Technically true, wrong impression
    "PRELIMINARY",            # Some evidence, too early
    "OUTDATED",               # Was true, evidence changed
    "NOT_SUPPORTED",          # No credible evidence
    "REFUTED",                # Directly contradicted
    "DANGEROUS",              # Could cause harm
]
```

## Development Commands

```bash
# Run tests
uv run pytest tests/ -v

# Run single agent test
uv run pytest tests/test_agents/test_decomposer.py -v

# Run Streamlit app
uv run streamlit run app/streamlit_app.py

# Run on specific claim
uv run python -m systems.s4_langgraph.system "Intermittent fasting reverses diabetes"

# Run baseline comparison
uv run python scripts/run_baselines.py --claim "Vitamin D prevents COVID"

# Run full evaluation
uv run python scripts/run_evaluation.py --dataset scifact --systems all
```

## Current Sprint Focus

1. **Week 1-2:** Project structure, API clients, basic PICO extraction, vaccine claims curation
2. **Week 3-4:** Core LangGraph pipeline end-to-end, guideline indexing, Simple RAG baseline (S2)
3. **Week 5-6:** Advanced RAG (S3), VLM + deep search (S6), alternative platform implementation (S5), Streamlit UI
4. **Week 7:** Run all 6 system variants on benchmarks, ablation studies, RAG tier comparison, agent platform comparison
5. **Week 8:** Results analysis, report writing, demo prep

## Code Style

- Python 3.12 (pinned via `.python-version`)
- Type hints everywhere
- Docstrings for all public functions
- Use Pydantic for data models
- Async where beneficial (API calls)
- Use `uv run` prefix for all commands

## Example Usage

```python
from systems.s4_langgraph.system import verify_claim

result = verify_claim("Intermittent fasting reverses Type 2 diabetes")

print(result["verdict"])           # "OVERSTATED"
print(result["confidence"])        # 0.78
print(result["explanation"])       # "While IF shows promise..."
print(result["sub_claims"])        # List of sub-claim verdicts
print(result["evidence"])          # Retrieved evidence with sources
print(result["agent_trace"])       # Execution trace for UI
```

## Retrieval Planner Logic

```python
def plan_retrieval(sub_claim: SubClaim, entities: Entities) -> RetrievalPlan:
    plan = RetrievalPlan()
    
    # Drug interaction? → DrugBank
    if entities.drugs and "interact" in sub_claim.text.lower():
        plan.add("drugbank_api")
    
    # Quantitative claim? → Need full-text deep search
    if sub_claim.has_numbers or sub_claim.asks_effect_size:
        plan.add("pubmed_api")
        plan.add("semantic_scholar")
        plan.needs_full_text = True
    
    # Asking about recommendations? → Guidelines
    if sub_claim.asks_recommendation:
        plan.add("guideline_store")
        plan.add("cochrane_api")
    
    # Has figures mentioned? → VLM
    if sub_claim.references_figure or plan.needs_full_text:
        plan.check_for_figures = True
    
    return plan
```

## Common Tasks

### Add a new retrieval source
1. Create client in `src/retrieval/new_source.py`
2. Add to `evidence_retriever.py` orchestration
3. Update `retrieval_planner.py` to use it
4. Add tests

### Add a new agent (S4/S6)
1. Create in `systems/s4_langgraph/agents/new_agent.py`
2. Update data models in `src/models.py` if needed
3. Add node in `systems/s4_langgraph/workflow.py`
4. Add evaluation in `src/evaluation/`

### Run on a single claim (debugging)
```python
from systems.s4_langgraph.system import verify_claim

result = await verify_claim("Vitamin D prevents COVID")
```

## Links

- [Full Proposal (v2)](./docs/Health_Claims_FactChecker_Proposal_v2.md)
- [SciFact Dataset](https://github.com/allenai/scifact)
- [PUBHEALTH Dataset](https://github.com/neemakot/Health-Fact-Checking)
- [PubMed E-utilities Docs](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [Semantic Scholar API](https://api.semanticscholar.org/)
- [LangGraph Docs](https://python.langchain.com/docs/langgraph)
