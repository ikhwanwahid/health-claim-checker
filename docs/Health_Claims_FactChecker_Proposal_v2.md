# Multi-Agent Fact-Checking for Clinical & Health Claims (v2)

## Combating Health Misinformation Through Evidence-Based Verification

---

## 1. The Concept

### One-Liner

> **Given a health claim, verify it against peer-reviewed research, clinical trials, and medical guidelines — using multi-stage retrieval from trusted sources — and produce a nuanced verdict with evidence and explanation.**

### Example

```
CLAIM: "Intermittent fasting reverses Type 2 diabetes"

SYSTEM OUTPUT:
┌──────────────────────────────────────────────────────────┐
│ Verdict: ⚠️ OVERSTATED                                   │
│ Confidence: 78%                                           │
│                                                           │
│ Sub-claims:                                               │
│ ① "Intermittent fasting affects Type 2 diabetes"          │
│   → ✓ SUPPORTED: Multiple RCTs show improved HbA1c       │
│     [PubMed API → 3 RCTs retrieved]                       │
│                                                           │
│ ② "The effect is reversal (not just improvement)"         │
│   → ⚠️ OVERSTATED: Liu et al., Lancet 2023 showed        │
│     remission in 47% at 3mo but only 20% at 12mo.        │
│     [Full-text retrieval → Results section, para 3]       │
│     [VLM → Figure 2: Kaplan-Meier remission curve]        │
│                                                           │
│ ③ "Evidence is conclusive"                                │
│   → ✗ INSUFFICIENT: Cochrane has no completed review.     │
│     WHO guidelines do not recommend IF for T2D.           │
│     [Cochrane API → no result]                            │
│     [Guideline search → WHO T2D guideline, Section 4.2]  │
│                                                           │
│ Overall: ⚠️ OVERSTATED                                    │
│ Explanation: While IF shows promise for glycemic control,  │
│ calling it "reversal" overstates the evidence.             │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Problem Statement

### Health Misinformation Is Everywhere

- Social media amplifies unverified health claims at scale
- Supplement companies make bold claims with cherry-picked studies
- News articles misrepresent study findings ("Coffee CURES cancer!")
- Patients make treatment decisions based on misinformation
- Even well-meaning health influencers oversimplify complex evidence

### The Nuance Problem

Health claims are rarely simply "true" or "false":

| Pattern | Example | Reality |
|---------|---------|---------|
| **Exaggerated** | "Vitamin D prevents COVID" | May reduce severity in deficient patients, not prevent infection |
| **Cherry-picked** | "Study proves X cures cancer" | One in-vitro study ≠ clinical proof |
| **Out of context** | "Red wine is good for your heart" | Observational correlation ≠ causation |
| **Outdated** | "Eggs are bad for cholesterol" | Guidelines revised |
| **Dose-blind** | "Turmeric fights inflammation" | True at lab concentrations, negligible at dietary doses |
| **Population-specific** | "Exercise lowers blood pressure" | Generally true, but varies by population |

### What's Needed

A system that can:
1. Decompose a complex health claim into verifiable sub-claims
2. Retrieve evidence from the **right source** using the **right method** per sub-claim
3. Read and extract data from paper figures (forest plots, survival curves)
4. Assess evidence quality (not all studies are equal)
5. Synthesize a nuanced verdict with explanation and citations

---

## 3. Core Design Insight: Multi-Method Retrieval

### RAG ≠ Vector Database

The common assumption is that "retrieval" means "embed everything into a vector DB and do cosine similarity search." This is wrong for health claims because:

1. **Medical literature has world-class search APIs** — PubMed's own BM25 + MeSH indexing beats any vector DB you'd build over the same content
2. **Evidence is distributed across source types** — papers, trial registries, guidelines, drug databases — no single index covers them all
3. **Different sub-claims need different retrieval strategies** — a drug interaction query needs a database lookup, not semantic search
4. **Pre-embedding 36M PubMed papers is unnecessary** — the API handles discovery; you only need deep search within the top candidates

### The Right Approach: Multi-Stage, Multi-Method Retrieval

```
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVAL METHOD BY STAGE                        │
│                                                               │
│  STAGE 1: DISCOVERY          "What papers exist?"             │
│  ──────────────────                                           │
│  • PubMed API         (BM25 + MeSH keyword search)    No DB  │
│  • Semantic Scholar    (citation-aware paper search)   No DB  │
│  • ClinicalTrials.gov (trial registry search)         No DB  │
│  • Cochrane API        (systematic review search)     No DB  │
│  → Result: ~30-50 candidate papers/documents                  │
│                                                               │
│  STAGE 2: RANKING            "Which are most relevant?"       │
│  ────────────────                                             │
│  • Cross-encoder re-ranker on abstracts               No DB  │
│  • Evidence hierarchy scoring (meta-analysis > RCT)   No DB  │
│  → Result: Top 10 most relevant, highest-quality papers       │
│                                                               │
│  STAGE 3: DEEP SEARCH        "Which exact passage?"           │
│  ────────────────────                                         │
│  • Download full-text from PMC (open access)                  │
│  • Chunk into sections (Methods, Results, Discussion)         │
│  • Embed chunks with PubMedBERT ← EMBEDDING HERE             │
│  • Retrieve specific passages per sub-claim                   │
│  + Search pre-indexed guidelines corpus ← EMBEDDING HERE      │
│  → Result: 5-10 specific passages with section references     │
│                                                               │
│  STAGE 4: STRUCTURED LOOKUP  "What does the database say?"    │
│  ──────────────────────────                                   │
│  • DrugBank API (drug info, interactions)             No DB   │
│  • FDA adverse events (safety data)                   No DB   │
│  • WHO/MOH guidelines (direct document fetch)         No DB   │
│  → Result: Structured facts (dosage, interactions, etc.)      │
│                                                               │
│  STAGE 5: VISUAL EXTRACTION  "What does the figure show?"     │
│  ──────────────────────────                                   │
│  • VLM reads charts from retrieved papers             No DB   │
│  → Result: Extracted numerical data from figures              │
│                                                               │
│  EMBEDDING IS USED IN 1 OF 5 STAGES                          │
│  (and only over ~200-500 chunks, not millions)                │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval Method by Claim Type

| Claim Type | Primary Retrieval | Embedding Needed? |
|-----------|-------------------|-------------------|
| "Does Drug X treat Condition Y?" | PubMed API → RCT abstracts | ❌ API sufficient |
| "What is the effect size (e.g., 20%)?" | PubMed API → full-text deep search | ✅ Find specific Results paragraph |
| "Does Drug X interact with Drug Y?" | DrugBank API / SQL | ❌ Structured lookup |
| "What does WHO recommend?" | Guideline corpus search | ✅ Pre-indexed guidelines |
| "What does Figure 3 in this paper show?" | VLM on rendered PDF page | ❌ Visual extraction |
| "Is this finding consistent across studies?" | PubMed API → multiple abstracts | ❌ API + LLM synthesis |
| "Has the evidence changed since 2020?" | PubMed API with date filter | ❌ API with temporal filter |

**Embedding plays a supporting role — the heavy lifting is in choosing the right retrieval method per sub-claim.**

---

## 4. Architecture

### Full System Flow

```
Health Claim Input
    │
    ▼
┌────────────────────────────────────────────┐
│   1. CLAIM DECOMPOSER                       │
│                                              │
│   Input:  "Intermittent fasting reverses     │
│            Type 2 diabetes"                  │
│                                              │
│   PICO Extraction:                           │
│     P (Population): Type 2 diabetes patients │
│     I (Intervention): Intermittent fasting   │
│     C (Comparison): Standard care / no IF    │
│     O (Outcome): Reversal / remission        │
│                                              │
│   Sub-claims:                                │
│     A: "IF has measurable effect on T2D"     │
│     B: "The effect is reversal (remission)"  │
│     C: "Evidence is conclusive"              │
│                                              │
│   Entities: drug=none, condition=T2DM,       │
│     intervention=IF, metric=HbA1c/remission  │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│   2. RETRIEVAL PLANNER                      │
│                                              │
│   Per sub-claim, decide WHICH retrieval      │
│   method(s) to use:                          │
│                                              │
│   Sub-claim A (effect exists?):              │
│     → PubMed API: "intermittent fasting      │
│       type 2 diabetes RCT"                   │
│     → Cochrane API: "intermittent fasting    │
│       diabetes"                              │
│                                              │
│   Sub-claim B (reversal specifically?):      │
│     → PubMed API: "intermittent fasting      │
│       diabetes remission"                    │
│     → Full-text deep search needed           │
│       (abstracts won't have remission rates) │
│                                              │
│   Sub-claim C (evidence conclusive?):        │
│     → Cochrane: check if systematic review   │
│       exists                                 │
│     → WHO guidelines: check for              │
│       recommendation                         │
│     → Evidence count + quality assessment    │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│   3. EVIDENCE RETRIEVAL (Multi-Method)                      │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  3a. API DISCOVERY (No embedding)                     │  │
│   │                                                        │  │
│   │  PubMed E-utilities:                                   │  │
│   │    search("intermittent fasting type 2 diabetes RCT")  │  │
│   │    → 45 papers (titles + abstracts + metadata)         │  │
│   │                                                        │  │
│   │  Semantic Scholar:                                     │  │
│   │    search("IF glycemic control randomized")            │  │
│   │    → 30 papers (with citation counts)                  │  │
│   │                                                        │  │
│   │  Cochrane Library:                                     │  │
│   │    search("intermittent fasting diabetes")             │  │
│   │    → 2 protocols, 0 completed reviews                  │  │
│   │                                                        │  │
│   │  ClinicalTrials.gov:                                   │  │
│   │    search("fasting diabetes", has_results=true)        │  │
│   │    → 8 completed trials with posted results            │  │
│   └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  3b. RE-RANKING (No embedding)                        │  │
│   │                                                        │  │
│   │  Cross-encoder scores each abstract against sub-claim  │  │
│   │  + Evidence hierarchy bonus:                           │  │
│   │    Systematic review: +0.3                             │  │
│   │    RCT: +0.2                                           │  │
│   │    Cohort: +0.1                                        │  │
│   │    Case report: +0.0                                   │  │
│   │                                                        │  │
│   │  → Top 10 papers selected                              │  │
│   └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  3c. DEEP SEARCH (Embedding — small, on-the-fly)      │  │
│   │                                                        │  │
│   │  For papers needing specific passage retrieval:        │  │
│   │    1. Download full text from PMC                      │  │
│   │    2. Chunk by section (~30 chunks per paper)          │  │
│   │    3. Embed chunks with PubMedBERT                     │  │
│   │    4. Search for sub-claim-specific passages           │  │
│   │                                                        │  │
│   │  Total embeddings: ~200-500 chunks (not millions)      │  │
│   │  Built per-claim, discarded after                      │  │
│   │                                                        │  │
│   │  + Search pre-indexed guideline corpus                 │  │
│   │    (WHO, MOH, NIH — ~500 docs, ~10K chunks, static)   │  │
│   └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  3d. STRUCTURED LOOKUP (No embedding)                 │  │
│   │                                                        │  │
│   │  DrugBank API: drug info, interactions                 │  │
│   │  ClinicalTrials.gov results: outcome measures          │  │
│   │  Direct document fetch: specific guideline by URL      │  │
│   └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────┐
│   4. VLM EXTRACTOR                          │
│                                              │
│   For papers with relevant figures:          │
│   • Render PDF pages containing figures      │
│   • Claude Vision extracts:                  │
│     - Forest plots: OR, CI, I² values        │
│     - Kaplan-Meier: survival/remission rates  │
│     - Bar charts: outcome values + error bars │
│     - CONSORT: enrollment/dropout numbers     │
│                                              │
│   Example output:                            │
│   "Figure 2, Liu et al. 2023:               │
│    Remission rate: 47.2% (IF) vs 2.8%       │
│    (control) at 3 months. At 12 months:     │
│    20.4% vs 1.2%"                            │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│   5. EVIDENCE GRADER                        │
│                                              │
│   For each piece of evidence, assess:        │
│                                              │
│   Study Design Score:                        │
│     Guidelines:          ★★★★★               │
│     Systematic reviews:  ★★★★                │
│     RCTs:                ★★★★                │
│     Cohort:              ★★★                 │
│     Case report:         ★★                  │
│     In-vitro:            ★                   │
│     Expert opinion:      ☆                   │
│                                              │
│   Quality Flags:                             │
│     • Sample size (n=15 vs n=5000)           │
│     • Funding source (industry vs NIH)       │
│     • Follow-up duration (3mo vs 5yr)        │
│     • Replication (single study vs many)     │
│                                              │
│   Consistency Check:                         │
│     • Do multiple studies agree?              │
│     • Is there a systematic review?           │
│     • Do guidelines reflect this evidence?    │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│   6. VERDICT AGENT                          │
│                                              │
│   Synthesize per sub-claim:                  │
│     A: ✓ SUPPORTED (3 RCTs, 1 meta-analysis)│
│     B: ⚠️ OVERSTATED (remission in 1 study,  │
│        not replicated, short follow-up)      │
│     C: ✗ INSUFFICIENT (no Cochrane review,   │
│        no WHO recommendation)                │
│                                              │
│   Overall: ⚠️ OVERSTATED                     │
│                                              │
│   Confidence: 78%                            │
│   Evidence quality: Moderate                 │
│   Plain-language explanation generated       │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│   7. SAFETY CHECKER                         │
│                                              │
│   Independent safety scan:                   │
│   • Is the claim about stopping medication?  │
│   • Could following this advice cause harm?  │
│   • Is it about a vulnerable population?     │
│                                              │
│   If dangerous: flag prominently + add       │
│   "Consult your doctor" warning              │
│                                              │
│   Runs REGARDLESS of verdict                 │
└────────────────────────────────────────────┘
```

### Agent Summary

| # | Agent | Type | Model | What It Does |
|---|-------|------|-------|-------------|
| 1 | **Claim Decomposer** | Function | Claude Sonnet | PICO extraction, sub-claim generation. Fixed steps — no reasoning loop needed. |
| 2 | **Retrieval Planner** | Agent (ReAct) | Claude Sonnet | Reasons about which retrieval methods to use per sub-claim. Uses tools to inspect entities, check claim type, and plan strategy. |
| 3 | **Evidence Retriever** | Agent (ReAct) | APIs + PubMedBERT | Searches for evidence using multiple tools (PubMed, S2, ClinicalTrials.gov, Cochrane, DrugBank). Reasons about whether enough evidence is found and adapts search strategy. |
| 4 | **VLM Extractor** | Agent (ReAct) | Claude Vision | Reasons about figure type (forest plot, Kaplan-Meier, bar chart), selects extraction strategy accordingly, and cross-checks figure data against abstract claims. |
| 5 | **Evidence Grader** | Agent (ReAct) | Claude Sonnet | Reasons about study quality — assesses design, sample size, bias indicators, funding conflicts. Compares contradicting evidence and explains discrepancies. |
| 6 | **Verdict Agent** | Agent (ReAct) | Claude Sonnet | Weighs evidence using hierarchy, compares claim language to evidence strength, synthesizes sub-claim verdicts into overall verdict with explanation. |
| 7 | **Safety Checker** | Function | Claude Sonnet | Pattern matching for dangerous claims. Fixed rules — no reasoning loop needed. |

**Architecture note:** 5 of 7 nodes are LangGraph ReAct agents (LLM + tools + reasoning loop). The Decomposer and Safety Checker remain fixed functions because their logic is deterministic. Each agent decides which tools to call, when to call them, and when it has enough information — similar to AWS Bedrock Agents or Strands, but orchestrated within LangGraph's state graph.

### New Agent: Retrieval Planner

This is a key addition — instead of blindly running every retrieval method, the agent **plans** which methods to use per sub-claim:

```python
# Pseudocode for Retrieval Planner
def plan_retrieval(sub_claims: list, entities: dict) -> list[RetrievalPlan]:
    plans = []
    for claim in sub_claims:
        plan = RetrievalPlan(claim=claim)
        
        # Does this need a drug lookup?
        if entities.get("drug"):
            plan.add(method="drugbank_api", query=entities["drug"])
        
        # Does this need research papers?
        if claim.needs_quantitative_evidence:
            plan.add(method="pubmed_api", query=build_pubmed_query(claim))
            plan.add(method="semantic_scholar", query=build_scholar_query(claim))
            plan.needs_full_text = True  # Will trigger deep search
        
        # Does this need guideline check?
        if claim.asks_about_recommendation:
            plan.add(method="guideline_search", query=claim.text)
            plan.add(method="cochrane_api", query=claim.text)
        
        # Does this need clinical trial results?
        if claim.mentions_trial_or_efficacy:
            plan.add(method="clinicaltrials_api", query=build_trial_query(claim))
        
        plans.append(plan)
    return plans
```

---

## 5. What Gets Embedded vs What Doesn't

### Honest Breakdown

| Component | Method | Embedding? | Size | Persistence |
|-----------|--------|-----------|------|-------------|
| **PubMed paper discovery** | E-utilities API (BM25 + MeSH) | ❌ | N/A | N/A |
| **Semantic Scholar discovery** | API (neural ranking) | ❌ | N/A | N/A |
| **Cochrane search** | Cochrane API | ❌ | N/A | N/A |
| **ClinicalTrials.gov** | API (keyword search) | ❌ | N/A | N/A |
| **Abstract re-ranking** | Cross-encoder model | ❌ | N/A | N/A |
| **Drug information** | DrugBank API / structured DB | ❌ | N/A | N/A |
| **Clinical guidelines** | Pre-indexed vector store | ✅ | ~10K chunks | Persistent (update quarterly) |
| **Full-text deep search** | On-the-fly embedding | ✅ | ~200-500 chunks | Ephemeral (per claim) |
| **Figure extraction** | VLM (Claude Vision) | ❌ | N/A | N/A |

### The Pre-Indexed Guideline Corpus (Only Persistent Vector DB)

| Source | Documents | Chunks | Update Frequency |
|--------|-----------|--------|-----------------|
| WHO clinical guidelines | ~50 | ~2,000 | Quarterly |
| NIH treatment guidelines | ~100 | ~3,000 | Quarterly |
| MOH Singapore guidelines | ~30 | ~1,000 | Quarterly |
| Cochrane plain-language summaries | ~8,000 | ~8,000 | Monthly |
| Drug reference summaries | ~2,000 | ~4,000 | Monthly |
| **Total** | **~10,000** | **~18,000** | — |

This is a small, focused vector DB — nothing like "embed all of PubMed."

### On-the-Fly Embedding (Per Claim, Ephemeral)

```python
# This happens ONLY when a sub-claim needs a specific passage
# from a full-text paper (e.g., exact remission rate in Results section)

def deep_search(papers: list, sub_claim: str) -> list[Passage]:
    chunks = []
    for paper in papers[:10]:
        full_text = download_pmc(paper.pmcid)
        if full_text is None:
            continue  # Not all papers are open access
        sections = split_by_section(full_text)
        chunks.extend(sections)  # ~30 chunks per paper
    
    # Small, temporary FAISS index
    embeddings = pubmedbert.encode([c.text for c in chunks])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    query_emb = pubmedbert.encode([sub_claim])
    scores, indices = index.search(query_emb, k=5)
    
    return [chunks[i] for i in indices[0]]
    # Index is discarded after this claim
```

---

## 6. Verdict Taxonomy

| Verdict | Definition | Example |
|---------|-----------|---------|
| **✓ SUPPORTED** | Strong evidence from high-quality studies | "Statins reduce cardiovascular events" |
| **✓ SUPPORTED (with caveats)** | True but requires context | "Exercise lowers BP" (varies by type) |
| **⚠️ OVERSTATED** | Kernel of truth but exaggerated | "IF reverses diabetes" (improves, not reverses) |
| **⚠️ MISLEADING** | Technically true but wrong impression | "Clinically tested" (but failed the test) |
| **⚠️ PRELIMINARY** | Some evidence, too early | "Psilocybin cures depression" (Phase 2 only) |
| **⚠️ OUTDATED** | Was true, evidence changed | "Eggs are bad for cholesterol" |
| **✗ NOT SUPPORTED** | No credible evidence | "Homeopathy cures cancer" |
| **✗ REFUTED** | Directly contradicted | "Vaccines cause autism" |
| **🚨 DANGEROUS** | Could cause harm if followed | "Stop insulin if you fast" |

---

## 7. Data Strategy

### 7.1 Evidence Sources (What We Search)

| Source | Access Method | Embedding? | Content |
|--------|-------------|-----------|---------|
| PubMed / MEDLINE | E-utilities API (free) | ❌ | 36M+ biomedical papers |
| Semantic Scholar | REST API (free) | ❌ | 200M+ papers with citations |
| ClinicalTrials.gov | REST API (free) | ❌ | 400K+ trials with results |
| Cochrane Library | API (free abstracts) | ❌ | Systematic reviews |
| PubMed Central | OA bulk download / API | ✅ (on-the-fly) | Full-text open-access papers |
| WHO / NIH / MOH Guidelines | Pre-downloaded + indexed | ✅ (persistent) | Clinical guidelines |
| DrugBank | API (free tier) | ❌ | Drug info, interactions |

### 7.2 Claims Dataset (What We Verify)

#### Existing Benchmarks (Free, Labeled)

| Dataset | Size | Domain | Labels | Link |
|---------|------|--------|--------|------|
| **SciFact** | 1,409 | Biomedical | Supports, Refutes, NEI | [github.com/allenai/scifact](https://github.com/allenai/scifact) |
| **PUBHEALTH** | 11,832 | Public health news | True, False, Mixture, Unproven | [github.com/neemakot/Health-Fact-Checking](https://github.com/neemakot/Health-Fact-Checking) |
| **HealthVer** | 14,330 | COVID health claims | Supports, Refutes, NEI | [github.com/sarrouti/HealthVer](https://github.com/sarrouti/HealthVer) |
| **COVID-Fact** | 4,086 | COVID-specific | True, False | [github.com/asaakyan/covidfact](https://github.com/asaakyan/covidfact) |
| **Total** | **31,657** | | | |

#### Label Mapping: Benchmark Labels → Our Verdicts

Our system produces **9 nuanced verdicts** but benchmark datasets use coarser labels (2–4 classes). To evaluate against benchmarks, we collapse our verdicts to match each dataset's label space:

| Our Verdict | SciFact / HealthVer (3) | PUBHEALTH (4) | COVID-Fact (2) |
|---|---|---|---|
| SUPPORTED | SUPPORTS | true | true |
| SUPPORTED_WITH_CAVEATS | SUPPORTS | mixture | true |
| OVERSTATED | NEI | mixture | false |
| MISLEADING | REFUTES | mixture | false |
| PRELIMINARY | NEI | unproven | false |
| OUTDATED | REFUTES | false | false |
| NOT_SUPPORTED | NEI | false | false |
| REFUTED | REFUTES | false | false |
| DANGEROUS | REFUTES | false | false |

**Key mapping decisions:**
- **OVERSTATED → NEI** (SciFact): A claim with a kernel of truth but exaggerated doesn't cleanly fit "supports" or "refutes" — NEI is the least-wrong bucket.
- **MISLEADING → REFUTES** (SciFact): Gives a wrong impression, closer to refutation than support.
- **SUPPORTED_WITH_CAVEATS → mixture** (PUBHEALTH): True but needs context aligns well with PUBHEALTH's "mixture" label.
- **PRELIMINARY → unproven** (PUBHEALTH): Direct semantic match — some evidence, too early to confirm.

**Information loss:** The collapse is lossy — PUBHEALTH preserves the most nuance (4 labels capture mixture/unproven), while COVID-Fact (2 labels) loses the most. This is expected and motivates our **dual evaluation strategy**:

1. **Benchmark evaluation** — Collapse verdicts, compute macro-F1 against dataset labels. Allows direct comparison with published baselines.
2. **Nuance evaluation** — Evaluate on our 200 curated claims with full 9-level ground truth. This measures what benchmarks cannot: can the system distinguish SUPPORTED from OVERSTATED, or PRELIMINARY from NOT_SUPPORTED?

The mapping utility lives in `src/evaluation/verdict_mapping.py`.

#### Custom Claims (~200, Singapore Context)

| Source | Examples |
|--------|---------|
| Singapore health news (CNA, ST) | "New study shows X prevents Y" |
| Health influencers | Supplement claims, diet claims |
| Traditional medicine | "TCM herb X treats condition Y" |
| Product marketing | "Clinically proven to improve Z" |
| MOH / HPB statements | Vaccination, screening claims |

#### Perturbation Augmentation

```
ORIGINAL (from Cochrane):
  "Metformin reduces HbA1c by 1.0% in T2D"  → SUPPORTED

PERTURBATIONS:
  "Metformin reduces HbA1c by 5.0%"          → OVERSTATED (wrong magnitude)
  "Metformin cures Type 2 diabetes"           → OVERSTATED (manage ≠ cure)
  "Metformin reduces HbA1c in Type 1"         → NOT SUPPORTED (wrong population)
  "Metformin is better than insulin"           → MISLEADING (depends on stage)
```

---

## 8. Evaluation Framework

### 8.1 Seven Evaluation Layers

| Layer | What | Metrics | Ground Truth Source |
|-------|------|---------|-------------------|
| **1. Decomposition** | Sub-claims correct? PICO extracted? | Sub-claim F1, PICO accuracy | Manual annotation (200 claims) |
| **2. Retrieval Planning** | Did planner choose correct methods per sub-claim? | Method selection accuracy | Manual annotation |
| **3. Evidence Retrieval** | Were relevant papers/passages found? | Recall@5, Recall@10, MRR | SciFact evidence labels |
| **4. VLM Extraction** | Were figure values correctly read? | Exact match, numeric ±5% | Human-annotated figures |
| **5. Evidence Grading** | Was study quality correctly assessed? | Agreement with expert ratings | Cochrane risk-of-bias data |
| **6. Verdict** | Was the final verdict correct? | Macro F1, per-class F1, calibration | Dataset labels |
| **7. Agent Tool Selection** | Did each agent call the right tools in the right order? | Tool selection precision, tool coverage recall | Manual annotation of expected tool sequences |

### 8.2 Baseline Comparison: Retrieval Strategy Study

This is the core contribution — comparing **retrieval strategies**, not just "RAG vs no-RAG":

| System | Retrieval Strategy | What It Tests |
|--------|-------------------|---------------|
| **B1: No retrieval** | LLM knowledge only | Is retrieval needed at all? |
| **B2: API-only RAG** | PubMed API → abstracts → LLM | Is simple API-based retrieval enough? |
| **B3: API + vector DB** | API discovery → embed all abstracts → cosine search → LLM | Does traditional vector RAG add value over API? |
| **S4: Multi-method** | API → cross-encoder re-rank → targeted deep search → LLM | Does stage-by-stage retrieval help? |
| **S5: Full pipeline** | Multi-method + VLM + Evidence Grader + Safety Checker | Does the full agent pipeline justify its complexity? |

### Why This Comparison Is More Interesting Than Standard RAG Studies

| Standard RAG Study | Our Study |
|-------------------|-----------|
| "Naive RAG vs Advanced RAG" | "When does API search suffice? When do you need embedding?" |
| Compares chunking strategies | Compares retrieval **methods** (API vs re-ranker vs embedding vs structured DB) |
| Single retrieval method | Agent decides which method per sub-claim |
| All evidence treated equally | Evidence hierarchy weights by study quality |

### 8.3 Ablation Studies

| Experiment | Hypothesis |
|-----------|-----------|
| API-only vs API + deep search | Deep search helps for quantitative claims but not for existence claims |
| With/without Retrieval Planner | Intelligent routing improves efficiency and accuracy vs running all methods |
| With/without cross-encoder re-ranking | Re-ranking improves precision (fewer irrelevant papers in context) |
| With/without evidence hierarchy | Weighting by study quality changes verdicts for contested claims |
| With/without VLM | Figure extraction resolves claims that text alone cannot |
| With/without Safety Checker | Dangerous claims are caught that Verdict Agent misses |
| PubMedBERT vs general embeddings | Domain-specific embeddings outperform general ones for deep search |
| Different LLMs as Verdict Agent | Claude vs GPT-4 vs open-weight for medical reasoning |
| Agent (ReAct) vs Function nodes | Does agent-style reasoning improve grading/verdict quality over fixed pipelines? |
| With/without agent tool selection | Letting agents choose tools vs hardcoded tool sequences — does flexibility help? |

### 8.4 Unique Evaluation Contributions

| Metric | What It Measures | Why It's Novel |
|--------|-----------------|----------------|
| **Retrieval method selection accuracy** | Did the planner pick the right method? | Nobody evaluates retrieval planning |
| **Evidence hierarchy adherence** | Does system prioritize meta-analyses over case reports? | Standard fact-checkers don't weight evidence |
| **Nuance detection rate** | Can it distinguish SUPPORTED from OVERSTATED? | Most benchmarks only have 3 labels |
| **Safety flag recall** | Does it catch ALL dangerous claims? | No existing health fact-checker evaluates this |
| **VLM medical figure accuracy** | Can VLM read forest plots and survival curves? | No systematic benchmark exists |
| **API vs embedding efficiency** | When does embedding add value over API? | Practical question rarely studied |
| **Agent tool selection accuracy** | Did each agent call the right tools for the claim type? | Generalizes retrieval planning eval across all agent nodes. Measures whether agents with reasoning loops make better tool choices than hardcoded sequences. Evaluated per-agent: e.g., did the Evidence Grader check for funding bias when the claim involves a pharmaceutical drug? |
| **Agent vs function comparison** | Does ReAct reasoning improve output quality over fixed pipelines? | Directly measures the value of agent autonomy — compares identical tools with vs without a reasoning loop |

---

## 9. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM (Agents)** | Claude Sonnet 4 | Strong medical reasoning, structured output |
| **VLM** | Claude Vision | Best at complex figure interpretation |
| **Cross-encoder** | ms-marco-MiniLM-L-12 or MedCPT | Re-ranking without embedding everything |
| **Embeddings** | PubMedBERT / BioSentVec | Only for deep search (~500 chunks) and guideline corpus |
| **Vector DB** | FAISS (in-memory) | Small corpus; no infrastructure needed |
| **Medical NER** | scispaCy (en_core_sci_lg) | Drug, condition, gene extraction |
| **MeSH mapping** | pymedtermino | Map terms to controlled vocabulary for PubMed queries |
| **APIs** | PubMed E-utilities, Semantic Scholar, ClinicalTrials.gov, DrugBank | Discovery-stage retrieval |
| **Agent Framework** | LangGraph | State management, conditional routing |
| **Observability** | Langfuse | Trace per-agent cost, latency, retrieval method usage |
| **UI** | Streamlit | Claim input + verdict display |

### API Integration Details

```python
# PubMed E-utilities (free, no key required for <3 req/sec)
from Bio import Entrez
Entrez.email = "team@example.com"
results = Entrez.esearch(db="pubmed", term="intermittent fasting diabetes RCT", retmax=20)

# Semantic Scholar (free, 100 req/5min)
import requests
r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
    params={"query": "intermittent fasting glycemic", "limit": 20,
            "fields": "title,abstract,citationCount,publicationTypes"})

# ClinicalTrials.gov (free, unlimited)
r = requests.get("https://clinicaltrials.gov/api/v2/studies",
    params={"query.cond": "diabetes", "query.intr": "fasting",
            "filter.overallStatus": "COMPLETED"})

# DrugBank (free tier)
# Structured JSON response with interactions, dosage, side effects

# Cochrane (PICO search)
# Search by Population + Intervention → returns systematic reviews
```

---

## 10. Directory Structure

```
health-claim-checker/
├── CLAUDE.md
├── README.md
├── requirements.txt
│
├── data/
│   ├── benchmarks/
│   │   ├── scifact/               # 1,409 claims
│   │   ├── pubhealth/             # 11,832 claims
│   │   └── healthver/             # 14,330 claims
│   ├── claims/
│   │   ├── curated_claims.json    # 200 custom claims
│   │   ├── perturbed_claims.json  # Augmented variants
│   │   └── ground_truth.json      # Labels + evidence
│   ├── guidelines/                # Pre-downloaded guideline PDFs
│   │   ├── who/
│   │   ├── nih/
│   │   └── moh_singapore/
│   └── figures/                   # Medical figures for VLM eval
│       ├── forest_plots/
│       ├── kaplan_meier/
│       └── ground_truth.json
│
├── src/
│   ├── agents/
│   │   ├── decomposer.py         # Claim → PICO + sub-claims
│   │   ├── retrieval_planner.py  # Decide method per sub-claim
│   │   ├── evidence_retriever.py # Orchestrate multi-method retrieval
│   │   ├── vlm_extractor.py      # Medical figure extraction
│   │   ├── evidence_grader.py    # Study quality + hierarchy
│   │   ├── verdict_agent.py      # Evidence → nuanced verdict
│   │   └── safety_checker.py     # Dangerous claim detection
│   │
│   ├── graph/
│   │   ├── state.py              # LangGraph state definition
│   │   └── workflow.py           # Agent orchestration graph
│   │
│   ├── retrieval/
│   │   ├── pubmed_client.py      # PubMed E-utilities wrapper
│   │   ├── semantic_scholar.py   # Semantic Scholar API
│   │   ├── cochrane_client.py    # Cochrane search
│   │   ├── clinical_trials.py    # ClinicalTrials.gov API
│   │   ├── drugbank_client.py    # Drug info API
│   │   ├── cross_encoder.py      # Abstract re-ranking
│   │   ├── deep_search.py        # On-the-fly full-text embedding
│   │   ├── guideline_store.py    # Pre-indexed guideline vector DB
│   │   └── trust_ranker.py       # Evidence hierarchy scoring
│   │
│   ├── medical_nlp/
│   │   ├── pico_extractor.py     # PICO element extraction
│   │   ├── medical_ner.py        # Drug, condition, gene NER
│   │   └── mesh_mapper.py        # Map to MeSH vocabulary
│   │
│   └── evaluation/
│       ├── decomposition_eval.py
│       ├── retrieval_planning_eval.py
│       ├── retrieval_eval.py
│       ├── vlm_eval.py
│       ├── grading_eval.py
│       ├── verdict_eval.py
│       └── safety_eval.py
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   ├── 01_api_exploration.ipynb       # Test PubMed, Semantic Scholar APIs
│   ├── 02_retrieval_comparison.ipynb  # API vs embedding vs hybrid
│   ├── 03_vlm_figure_extraction.ipynb # Test VLM on medical figures
│   ├── 04_scifact_baseline.ipynb      # Run SciFact benchmark
│   └── 05_results_analysis.ipynb
│
└── scripts/
    ├── download_benchmarks.py
    ├── index_guidelines.py        # One-time: build guideline vector DB
    ├── extract_figures.py         # Extract figures from papers for VLM eval
    ├── run_baselines.py
    └── run_evaluation.py
```

---

## 11. Workload Division (6 People)

| Track | Member | Responsibilities |
|-------|--------|-----------------|
| **DATA & CLAIMS** | Member 1 | Download SciFact/PUBHEALTH/HealthVer. Curate 200 Singapore health claims. Create perturbation variants. Extract medical figures for VLM benchmark. Build ground truth annotations. |
| **API RETRIEVAL** | Member 2 | PubMed E-utilities integration. Semantic Scholar API. ClinicalTrials.gov API. Cochrane search. DrugBank integration. Rate limiting, caching, error handling. |
| **RANKING & DEEP SEARCH** | Member 3 | Cross-encoder re-ranker for abstracts. Evidence hierarchy scoring. On-the-fly full-text embedding with PubMedBERT. Pre-indexed guideline vector store. Trust-based ranking. |
| **AGENTS & ORCHESTRATION** | Member 4 | LangGraph workflow. Claim Decomposer with PICO. Retrieval Planner agent. Orchestrator routing logic. State management. scispaCy NER + MeSH mapping. |
| **VLM & VERDICT** | Member 5 | VLM Extractor for forest plots, Kaplan-Meier, bar charts. Evidence Grader (GRADE framework). Verdict Agent with 9-level taxonomy. Safety Checker. Streamlit UI. |
| **EVALUATION** | Member 6 | Evaluation pipeline (6 layers). All 5 baseline implementations. Ablation study runner. Langfuse tracing. SciFact/PUBHEALTH benchmarks. Retrieval method comparison analysis. Results visualization. |

### Why This Split Works Better

| Track | What They Learn | Portfolio Value |
|-------|----------------|----------------|
| DATA & CLAIMS | Benchmark curation, data augmentation | Data engineering |
| API RETRIEVAL | External API integration, rate limiting | Backend / integration engineering |
| RANKING & DEEP SEARCH | Re-ranking, embedding, hybrid retrieval | ML engineering / search |
| AGENTS & ORCHESTRATION | LangGraph, multi-agent design, NER | AI engineering |
| VLM & VERDICT | Vision models, classification, UX | ML + product |
| EVALUATION | Benchmarking, ablation studies, analysis | Research / ML ops |

Each person works with a **different retrieval paradigm** — API, re-ranking, embedding, VLM — which makes the team's skill coverage much broader.

---

## 12. Timeline (8-Week Sprint)

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| **1-2** | **Data & APIs** | Benchmarks downloaded. PubMed, Semantic Scholar, ClinicalTrials.gov APIs working. Basic PICO extraction. 50 custom claims curated. Guideline corpus downloaded. |
| **3-4** | **Core Pipeline** | Decomposer + Retrieval Planner + API retrieval + re-ranking working end-to-end. Guideline vector store built. Run on SciFact subset for early signal. |
| **5-6** | **Advanced Features** | Full-text deep search. VLM figure extraction. Evidence Grader. Safety Checker. Full multi-method retrieval. Streamlit UI. |
| **7** | **Evaluation** | All 5 baselines run. Ablation studies. SciFact + PUBHEALTH benchmarks. Custom claim evaluation. Retrieval method comparison. |
| **8** | **Polish** | Report writing. Demo prep. Code cleanup. Reproducibility check. Results visualization. |

---

## 13. Risks & Guardrails

| Risk | Mitigation |
|------|-----------|
| **False "SUPPORTED" on dangerous claim** | Safety Checker runs independently; dangerous claims always flagged |
| **System used as medical advice** | Disclaimer on every output: "Not medical advice. Consult a healthcare professional." |
| **Hallucinated citations** | All PubMed IDs verified via API — if PMID doesn't exist, citation is rejected |
| **Rate limiting on APIs** | Implement caching layer; PubMed allows 10 req/sec with API key |
| **Papers behind paywalls** | Only use open-access full text (PMC); fall back to abstract if no full text |
| **Outdated evidence** | Prioritize recent publications; note publication date in verdict |
| **Population-specific claims** | PICO extraction captures population; verdict notes limitations |

---

## 14. Course Requirements Mapping

| Requirement | How Met |
|-------------|---------|
| **Real-world problem** | ✅ Health misinformation — WHO-recognized global threat |
| **Substantial GenAI** | ✅ 7 agents + multi-method retrieval + VLM + medical NER |
| **Quantitative metrics** | ✅ Macro F1, Recall@k, VLM accuracy, calibration, safety recall |
| **Compare 2+ alternatives** | ✅ 5 retrieval strategies compared + 8 ablation studies |
| **Reflective analysis** | ✅ "When does API search suffice vs when do you need embedding?" |
| **Risks & mitigations** | ✅ 7 risks with specific guardrails + safety-first design |
| **Reproducibility** | ✅ Public benchmarks (30K+ claims), free APIs, evaluation scripts |

### Core Research Question

> **"For multi-source health claim verification, when does API-based retrieval suffice, when do you need dense retrieval, and how should an agent decide between them?"**

This is a more nuanced and interesting question than "does RAG help?" — and the 5-system comparison directly answers it.

---

## 15. The Elevator Pitch (30 seconds)

> "Health misinformation kills people. Our system takes any health claim — 'Intermittent fasting reverses diabetes', 'Vitamin D prevents COVID' — and verifies it using a multi-agent pipeline. Unlike traditional RAG systems that embed everything into a vector database, our agents intelligently choose the right retrieval method per claim: PubMed API for paper discovery, cross-encoder re-ranking for relevance, targeted embedding only for deep within-paper search, and vision models to extract data from medical figures like forest plots and survival curves. We evaluate on 30,000+ existing benchmark claims and compare five retrieval strategies to answer: when does API search suffice, and when do you actually need embedding? The system produces nuanced verdicts — not just true or false, but 'supported', 'overstated', 'preliminary', or 'dangerous' — with full citations and plain-language explanation."
