# S4/S6: LangGraph Multi-Agent Pipeline

LangGraph-based multi-agent system for health claim verification. S4 and S6 share this codebase — a config flag enables VLM extraction and deep search for S6 (full agentic RAG).

## Architecture

```
Claim → Decomposer → Retrieval Planner → Evidence Retriever → VLM Extractor
                                                    ↓
                              Verdict Agent ← Evidence Grader ← Safety Checker
```

## 7 Pipeline Nodes

| Node | Type | Purpose | Status |
|------|------|---------|--------|
| Claim Decomposer | Function | PICO extraction + sub-claim decomposition | Done |
| Retrieval Planner | Agent (ReAct) | Decide retrieval methods per sub-claim | Done |
| Evidence Retriever | Agent (ReAct) | Multi-method search with adaptive strategy | Not started |
| VLM Extractor | Agent (ReAct) | Medical figure data extraction (S6 only) | Not started |
| Evidence Grader | Agent (ReAct) | Study quality + hierarchy scoring | Not started |
| Verdict Agent | Agent (ReAct) | Weigh evidence, synthesize verdict | Not started |
| Safety Checker | Function | Flag dangerous claims | Not started |

## Multi-Method Retrieval

The evidence retriever orchestrates multiple retrieval methods. Not all methods use embeddings — this is a core design insight.

| Stage | Method | Embedding? | Status |
|-------|--------|-----------|--------|
| Discovery | PubMed API | No | Done |
| Discovery | Semantic Scholar | No | Done |
| Discovery | Cochrane API | No | Done |
| Discovery | ClinicalTrials.gov | No | Done |
| Discovery | OpenFDA / RxNorm | No | Done |
| Ranking | Cross-encoder re-ranker | No | Done |
| Deep Search | PubMedBERT full-text chunk embedding (S6) | Yes (small, ephemeral) | Not started |
| Guidelines | Pre-indexed guideline vector DB | Yes (persistent) | Not started |
| Visual | Claude Vision on PDF figures (S6) | No | Not started |

## S4 vs S6

Both variants use this same pipeline. The difference is config-driven:

| Feature | S4 | S6 |
|---------|----|----|
| PICO decomposition | Yes | Yes |
| API discovery + reranking | Yes | Yes |
| Deep search (full-text PubMedBERT) | No | Yes |
| VLM figure extraction | No | Yes |

## To-Do

- [ ] **Evidence Retriever agent** — orchestrate multi-method search using the retrieval plan
- [ ] **Evidence Grader agent** — GRADE framework quality scoring, bias assessment
- [ ] **Verdict Agent** — synthesize evidence into 9-level verdict with confidence score
- [ ] **Safety Checker** — flag dangerous claims (pattern matching + LLM)
- [ ] **VLM Extractor agent** — Claude Vision for forest plots, Kaplan-Meier curves (S6)
- [ ] **Deep search integration** — PubMedBERT on-the-fly embedding (S6)

## Key Files

```
systems/s4_langgraph/
├── system.py           # verify_claim() entry point
├── workflow.py         # LangGraph state graph (7 nodes wired, compiles)
└── agents/
    ├── retrieval_planner.py    # Done — ReAct agent with 3 tools
    ├── evidence_retriever.py   # Stub
    ├── vlm_extractor.py        # Stub
    ├── evidence_grader.py      # Stub
    └── verdict_agent.py        # Stub
```
