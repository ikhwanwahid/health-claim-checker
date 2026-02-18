# Health Claims Fact-Checker

Multi-agent system for verifying health claims against peer-reviewed research, clinical trials, and medical guidelines.

Uses **multi-method retrieval** (not just vector DB) — agents intelligently choose the right retrieval method per sub-claim: API search, cross-encoder re-ranking, targeted embedding, and VLM.

For full project details, architecture rationale, and evaluation plan, see the [project proposal](docs/Health_Claims_FactChecker_Proposal_v2.md).

## Setup

### Prerequisites

- Python 3.12 (pinned via `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd health-claim-checker

# Install dependencies
uv sync

# Install scispaCy biomedical model
uv pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Copy env template and add your API keys
cp .env.example .env
```

### API Keys

Edit `.env` with your keys:

```bash
# Required — powers all Claude-based agents
ANTHROPIC_API_KEY=sk-ant-...

# Optional — higher rate limits for PubMed and Semantic Scholar
NCBI_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
```

Without `ANTHROPIC_API_KEY`, the decomposer and retrieval planner fall back to rule-based mode (no LLM calls). All retrieval APIs (PubMed, Semantic Scholar, ClinicalTrials.gov, Cochrane, OpenFDA) work without any key.

### Jupyter Kernel

To run the notebooks, register the project kernel:

```bash
uv run python -m ipykernel install --user --name health-claim-checker --display-name "Health Claim Checker (Python 3.12)"
```

Then select **"Health Claim Checker (Python 3.12)"** as the kernel in Jupyter.

To launch:

```bash
uv run jupyter notebook notebooks/
```

## Running

```bash
# Run tests
uv run pytest tests/ -v

# Run the Streamlit app (UI is scaffolded, pipeline partially functional)
uv run streamlit run app/streamlit_app.py
```

## Comparison Framework

This project compares health claim verification across two dimensions:

### Thread 1: RAG Tiers — When does retrieval sophistication add value?

| Tier | What It Does | Embedding? |
|------|-------------|-----------|
| **Simple RAG** | Embed vaccine articles → vector search → LLM verdict | Yes (static corpus) |
| **Advanced RAG** | PICO query reformulation → Multi-source API → Cross-encoder rerank → LLM verdict | Minimal (reranker only) |
| **Agentic RAG** | ReAct agent → API discovery → rerank → deep search → VLM → grading → verdict | Yes (ephemeral, targeted) |

### Thread 2: Agent Architectures — Does the framework matter?

Same pipeline implemented on multiple platforms. Function pipeline included as no-agent baseline.

| Platform | Style |
|----------|-------|
| **LangGraph** | Structured state graph, typed state, conditional routing |
| **Alt Framework** | Team picks 1–2 alternatives (CrewAI / AutoGen / smolagents / PydanticAI) |
| **Function Pipeline** | Plain Python functions, no agent reasoning |

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

Each variant exposes `verify_claim(claim) → FactCheckResult` — the evaluation framework runs any variant interchangeably. See [`systems/README.md`](systems/README.md) for the shared output contract and [`systems/s4_langgraph/README.md`](systems/s4_langgraph/README.md) for the S4/S6 pipeline architecture.

## Project Structure

```
health-claim-checker/
├── src/                               # SHARED LIBRARY — all variants import from here
│   ├── config.py                      #   Settings, API keys, model configs
│   ├── models.py                      #   Done — FactCheckState, SubClaim, Evidence, etc.
│   ├── functions/                     #   Single-pass nodes (no reasoning loop)
│   │   ├── decomposer.py             #   Done — PICO + sub-claims
│   │   └── safety_checker.py         #   Stub
│   ├── retrieval/
│   │   ├── pubmed_client.py           #   Done
│   │   ├── semantic_scholar.py        #   Done
│   │   ├── cochrane_client.py         #   Done
│   │   ├── clinical_trials.py         #   Done
│   │   ├── drugbank_client.py         #   Done
│   │   ├── cross_encoder.py           #   Done — re-rank abstracts by relevance
│   │   ├── deep_search.py             #   Not started
│   │   ├── guideline_store.py         #   Not started
│   │   └── trust_ranker.py            #   Not started
│   ├── medical_nlp/
│   │   ├── medical_ner.py             #   Done — scispaCy entity extraction
│   │   ├── pico_extractor.py          #   Done — LLM + rule-based
│   │   └── mesh_mapper.py             #   Done — MeSH vocabulary mapping
│   └── evaluation/                    #   Not started (all 8 eval modules)
│
├── systems/                           # VARIANT IMPLEMENTATIONS — symmetric, independent
│   ├── README.md                      #   Shared output contract
│   ├── s1_no_retrieval/               #   S1: LLM-only baseline — stub
│   ├── s2_simple_rag/                 #   S2: Static corpus + vector search — stub
│   ├── s3_advanced_rag/               #   S3: Function pipeline — stub
│   ├── s4_langgraph/                  #   S4 + S6: LangGraph multi-agent (config-driven)
│   │   ├── agents/                    #   ReAct agents (retrieval_planner done, rest stubs)
│   │   ├── workflow.py                #   Done — LangGraph orchestration (7 nodes wired)
│   │   └── system.py                  #   verify_claim() entry point
│   └── s5_alt_platform/               #   S5: Alternative agent framework — stub
│
├── app/
│   └── streamlit_app.py               #   Scaffolded — UI layout done, pipeline integration pending
├── notebooks/
│   ├── 01_api_exploration.ipynb       #   Done — tests all retrieval APIs + medical NLP
│   └── 02_retrieval_comparison.ipynb  #   Done — retrieval planner + PICO evaluation (30 claims, 3 metrics)
├── tests/
│   ├── test_agents/
│   │   └── test_retrieval_planner.py  #   Done — 44 tests
│   └── test_retrieval/
│       └── test_cross_encoder.py      #   Done — 17 tests
├── data/
│   ├── benchmarks/                    #   Placeholder dirs (download script scaffolded)
│   ├── guidelines/                    #   Empty dirs (who/, nih/, moh_singapore/)
│   ├── claims/
│   │   └── pico_ground_truth.json     #   Done — 30 hand-labeled claims for PICO eval
│   └── figures/                       #   Not started
└── scripts/
    └── download_benchmarks.py         #   Scaffolded — creates dirs, no actual downloads
```

## What's Done

### Shared Infrastructure
- [x] **Data models** — `src/models.py` (FactCheckState, SubClaim, Evidence, AgentTrace, ToolCall)
- [x] **Config** — env-based API key loading
- [x] **Streamlit app** — scaffolded multi-tab UI

### Retrieval Clients (`src/retrieval/`)
- [x] PubMed E-utilities (search, fetch, PICO query builder)
- [x] Semantic Scholar (search, paper fetch, citations, TLDRs)
- [x] Cochrane (systematic review search via PubMed + S2)
- [x] ClinicalTrials.gov (search, filter by status/phase/type)
- [x] OpenFDA + RxNorm (drug labels, interactions)
- [x] Cross-encoder re-ranker (ms-marco-MiniLM, sigmoid normalization, graceful fallback)

### Medical NLP (`src/medical_nlp/`)
- [x] Entity extraction (scispaCy with classification heuristics)
- [x] PICO extraction (dual-mode: LLM + rule-based, comparison hallucination fix applied)
- [x] PICO evaluation (30 hand-labeled claims, 3 metrics: Jaccard, Token F1, LLM Judge)
- [x] MeSH vocabulary mapping (NCBI E-utilities)

### System Variants
- [x] **S4/S6 LangGraph** — workflow graph wired (7 nodes compile), retrieval planner agent done. See [`systems/s4_langgraph/README.md`](systems/s4_langgraph/README.md) for pipeline details.
- [ ] S1–S3, S5 — stubs only

### Notebooks
- [x] `01_api_exploration` — end-to-end test of all retrieval APIs and NLP modules
- [x] `02_retrieval_comparison` — retrieval planner testing + PICO extraction evaluation (rule-based vs LLM, 88% LLM accuracy)

### Tests
- [x] Retrieval planner (44 tests)
- [x] Cross-encoder re-ranker (17 tests)

## To-Do

### Shared Retrieval Infrastructure
- [ ] **Deep search** — PubMedBERT on-the-fly full-text chunk embedding (~500 chunks/claim)
- [ ] **Guideline store** — pre-indexed vector DB for WHO/NIH/MOH guidelines
- [ ] **Trust ranker** — evidence hierarchy scoring (guideline > systematic review > RCT > ...)
- [ ] **Index guidelines script** — download and index guideline PDFs

### System Variants
- [ ] **S1** — LLM-only baseline implementation
- [ ] **S2** — Simple RAG (static corpus, vector search)
- [ ] **S3** — Advanced RAG function pipeline
- [ ] **S4/S6** — remaining agents (evidence retriever, VLM, grader, verdict, safety). See [`systems/s4_langgraph/README.md`](systems/s4_langgraph/README.md).
- [ ] **S5** — alternative agent framework implementation

### Evaluation
- [ ] Decomposition eval
- [ ] Retrieval planning eval (tool selection accuracy)
- [ ] Retrieval eval (recall, precision)
- [ ] VLM eval (figure extraction accuracy)
- [ ] Grading eval
- [ ] Verdict eval
- [ ] Safety eval
- [ ] Download benchmarks script (SciFact, PUBHEALTH, HealthVer)
- [ ] Run baselines script
- [ ] Run evaluation script

### Polish
- [ ] Streamlit app pipeline integration
- [ ] `03_vlm_figure_extraction.ipynb`
- [ ] `04_scifact_baseline.ipynb`
- [ ] `05_results_analysis.ipynb`
- [x] PICO ground truth data (30 hand-labeled claims)
- [ ] Curated claims + ground truth data (other benchmarks)
- [ ] Tests for decomposer, retrieval clients, evaluation modules

## Evidence Hierarchy

```python
EVIDENCE_WEIGHTS = {
    "guideline": 1.0,           # WHO, NIH, MOH
    "systematic_review": 0.9,   # Cochrane, meta-analyses
    "rct": 0.8,                 # Randomized controlled trials
    "cohort": 0.6,
    "case_control": 0.5,
    "case_report": 0.3,
    "in_vitro": 0.2,
    "expert_opinion": 0.1,
}
```

## Verdict Taxonomy

`SUPPORTED` | `SUPPORTED_WITH_CAVEATS` | `OVERSTATED` | `MISLEADING` | `PRELIMINARY` | `OUTDATED` | `NOT_SUPPORTED` | `REFUTED` | `DANGEROUS`
