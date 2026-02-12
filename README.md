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

## Architecture

```
Claim → Decomposer → Retrieval Planner → Evidence Retriever → VLM Extractor
                                                    ↓
                              Verdict Agent ← Evidence Grader ← Safety Checker
```

### 7 Pipeline Nodes

| Node | Type | Purpose | Status |
|------|------|---------|--------|
| Claim Decomposer | Function | PICO extraction + sub-claim decomposition | Done |
| Retrieval Planner | Agent (ReAct) | Decide retrieval methods per sub-claim | Done |
| Evidence Retriever | Agent (ReAct) | Multi-method search with adaptive strategy | Not started |
| VLM Extractor | Agent (ReAct) | Medical figure data extraction | Not started |
| Evidence Grader | Agent (ReAct) | Study quality + hierarchy scoring | Not started |
| Verdict Agent | Agent (ReAct) | Weigh evidence, synthesize verdict | Not started |
| Safety Checker | Function | Flag dangerous claims | Not started |

### Multi-Method Retrieval

| Method | What it does | Status |
|--------|-------------|--------|
| PubMed API | Search biomedical literature | Done |
| Semantic Scholar | Academic paper search with citations | Done |
| Cochrane API | Systematic review search | Done |
| ClinicalTrials.gov | Registered trial search | Done |
| OpenFDA / RxNorm | Drug labels and interactions | Done |
| Cross-encoder | Re-rank abstracts by relevance | Not started |
| Deep search | PubMedBERT full-text chunk embedding | Not started |
| Guideline store | Pre-indexed clinical guideline vector DB | Not started |

## Project Structure

```
health-claim-checker/
├── src/
│   ├── config.py                  # Settings, API keys, model configs
│   ├── agents/                    # ReAct agents (LLM + tools + reasoning loop)
│   │   ├── retrieval_planner.py   #   Done — decide method per sub-claim
│   │   ├── evidence_retriever.py  #   Stub
│   │   ├── vlm_extractor.py       #   Stub
│   │   ├── evidence_grader.py     #   Stub
│   │   └── verdict_agent.py       #   Stub
│   ├── functions/                 # Single-pass nodes (no reasoning loop)
│   │   ├── decomposer.py          #   Done — PICO + sub-claims
│   │   └── safety_checker.py      #   Stub
│   ├── graph/
│   │   ├── state.py               #   Done — FactCheckState, SubClaim, Evidence, etc.
│   │   └── workflow.py            #   Done — LangGraph orchestration (7 nodes wired)
│   ├── retrieval/
│   │   ├── pubmed_client.py       #   Done
│   │   ├── semantic_scholar.py    #   Done
│   │   ├── cochrane_client.py     #   Done
│   │   ├── clinical_trials.py     #   Done
│   │   ├── drugbank_client.py     #   Done
│   │   ├── cross_encoder.py       #   Not started
│   │   ├── deep_search.py         #   Not started
│   │   ├── guideline_store.py     #   Not started
│   │   └── trust_ranker.py        #   Not started
│   ├── medical_nlp/
│   │   ├── medical_ner.py         #   Done — scispaCy entity extraction
│   │   ├── pico_extractor.py      #   Done — LLM + rule-based
│   │   └── mesh_mapper.py         #   Done — MeSH vocabulary mapping
│   └── evaluation/                #   Not started (all 8 eval modules)
├── app/
│   └── streamlit_app.py           #   Scaffolded — UI layout done, pipeline integration pending
├── notebooks/
│   ├── 01_api_exploration.ipynb   #   Done — tests all retrieval APIs + medical NLP
│   └── 02_retrieval_comparison.ipynb  # Done — retrieval planner + PICO evaluation (30 claims, 3 metrics)
├── tests/
│   └── test_agents/
│       └── test_retrieval_planner.py  # Done — 44 tests
├── data/
│   ├── benchmarks/                #   Placeholder dirs (download script scaffolded)
│   ├── guidelines/                #   Empty dirs (who/, nih/, moh_singapore/)
│   ├── claims/
│   │   └── pico_ground_truth.json #   Done — 30 hand-labeled claims for PICO eval
│   └── figures/                   #   Not started
└── scripts/
    └── download_benchmarks.py     #   Scaffolded — creates dirs, no actual downloads
```

## What's Done

### Pipeline Nodes
- [x] **Claim Decomposer** — scispaCy NER, PICO extraction (LLM + rule-based), sub-claim decomposition
- [x] **Retrieval Planner** — ReAct agent with 3 tools, rule-based fallback, validated against 8 test claims with discrimination checks (4/4 passing)

### Retrieval Clients
- [x] PubMed E-utilities (search, fetch, PICO query builder)
- [x] Semantic Scholar (search, paper fetch, citations, TLDRs)
- [x] Cochrane (systematic review search via PubMed + S2)
- [x] ClinicalTrials.gov (search, filter by status/phase/type)
- [x] OpenFDA + RxNorm (drug labels, interactions)

### Medical NLP
- [x] Entity extraction (scispaCy with classification heuristics)
- [x] PICO extraction (dual-mode: LLM + rule-based, comparison hallucination fix applied)
- [x] PICO evaluation (30 hand-labeled claims, 3 metrics: Jaccard, Token F1, LLM Judge)
- [x] MeSH vocabulary mapping (NCBI E-utilities)

### Infrastructure
- [x] LangGraph state definitions (FactCheckState, SubClaim, Evidence, AgentTrace, ToolCall)
- [x] Workflow graph (all 7 nodes wired, compiles)
- [x] Config with env-based API key loading
- [x] Streamlit app scaffolding (multi-tab UI)

### Notebooks
- [x] `01_api_exploration` — end-to-end test of all retrieval APIs and NLP modules
- [x] `02_retrieval_comparison` — retrieval planner testing + PICO extraction evaluation (rule-based vs LLM, 88% LLM accuracy)

### Tests
- [x] Retrieval planner (44 tests — tools, rule-based fallback, ReAct loop, integration)

## To-Do

### Critical Path (blocks end-to-end pipeline)
- [ ] **Evidence Retriever agent** — orchestrate multi-method search using the retrieval plan
- [ ] **Cross-encoder re-ranker** — re-rank retrieved abstracts by relevance to sub-claim
- [ ] **Evidence Grader agent** — GRADE framework quality scoring, bias assessment
- [ ] **Verdict Agent** — synthesize evidence into 9-level verdict with confidence score
- [ ] **Safety Checker** — flag dangerous claims (pattern matching + LLM)

### Retrieval Infrastructure
- [ ] **Deep search** — PubMedBERT on-the-fly full-text chunk embedding (~500 chunks/claim)
- [ ] **Guideline store** — pre-indexed vector DB for WHO/NIH/MOH guidelines
- [ ] **Trust ranker** — evidence hierarchy scoring (guideline > systematic review > RCT > ...)
- [ ] **Index guidelines script** — download and index guideline PDFs

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

### VLM Pipeline
- [ ] **VLM Extractor agent** — Claude Vision for forest plots, Kaplan-Meier curves
- [ ] Figure extraction script
- [ ] `03_vlm_figure_extraction.ipynb`

### Polish
- [ ] Streamlit app pipeline integration
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
