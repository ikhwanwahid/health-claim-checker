"""End-to-end demo: full pipeline through Verdict Agent + Safety Checker.

Runs all 6 pipeline nodes on a claim and prints detailed output
at each step so you can see what each node does.

Usage:
    uv run python scripts/e2e_verdict_demo.py
    uv run python scripts/e2e_verdict_demo.py "Your custom claim here"
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

from src.functions.decomposer import run_decomposer
from src.functions.safety_checker import run_safety_checker
from src.models import FactCheckState
from systems.s4_langgraph.agents.retrieval_planner import run_retrieval_planner
from systems.s4_langgraph.agents.evidence_retriever import run_evidence_retriever
from systems.s4_langgraph.agents.evidence_grader import run_evidence_grader
from systems.s4_langgraph.agents.verdict_agent import run_verdict_agent


DEFAULT_CLAIM = "The MMR vaccine causes autism in children"


def _hr(title: str = "") -> None:
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}\n")
    else:
        print(f"\n{'-' * 70}\n")


def _wrap(text: str, indent: int = 4) -> str:
    return textwrap.fill(text, width=80, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


def print_step1(state: FactCheckState) -> None:
    """Print decomposer output."""
    _hr("STEP 1: Claim Decomposer (function node)")
    print("What it does:")
    print("  1. Extracts medical entities via scispaCy NER")
    print("  2. Extracts PICO framework elements")
    print("  3. Decomposes the claim into atomic sub-claims")
    _hr()

    entities = state.get("entities", {})
    print("  Extracted entities:")
    for etype, elist in entities.items():
        if elist:
            print(f"    {etype}: {', '.join(elist)}")
    if not any(v for v in entities.values()):
        print("    (none detected)")

    pico = state.get("pico")
    print(f"\n  PICO extraction:")
    if pico:
        print(f"    Population:   {pico.population or '—'}")
        print(f"    Intervention: {pico.intervention or '—'}")
        print(f"    Comparison:   {pico.comparison or '—'}")
        print(f"    Outcome:      {pico.outcome or '—'}")
    else:
        print("    (no PICO)")

    sub_claims = state.get("sub_claims", [])
    print(f"\n  Sub-claims ({len(sub_claims)}):")
    for sc in sub_claims:
        print(f"    [{sc.id}] \"{sc.text}\"")
        if sc.pico:
            parts = []
            if sc.pico.population:
                parts.append(f"P={sc.pico.population}")
            if sc.pico.intervention:
                parts.append(f"I={sc.pico.intervention}")
            if sc.pico.comparison:
                parts.append(f"C={sc.pico.comparison}")
            if sc.pico.outcome:
                parts.append(f"O={sc.pico.outcome}")
            if parts:
                print(f"           PICO: {', '.join(parts)}")

    trace = state["agent_trace"][-1]
    print(f"\n  Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_step2(state: FactCheckState) -> None:
    """Print retrieval planner output."""
    _hr("STEP 2: Retrieval Planner (ReAct agent)")
    print("What it does:")
    print("  Examines each sub-claim's characteristics and decides")
    print("  which retrieval methods to use. Methods include:")
    print("    pubmed_api, semantic_scholar, cochrane_api,")
    print("    clinical_trials, drugbank_api, cross_encoder,")
    print("    deep_search, guideline_store")
    _hr()

    plan = state.get("retrieval_plan", {})
    for sc_id, methods in plan.items():
        sc = next((s for s in state["sub_claims"] if s.id == sc_id), None)
        label = f"\"{sc.text}\"" if sc else sc_id
        print(f"  [{sc_id}] {label}")
        print(f"    → Methods: {', '.join(methods)}")

    trace = state["agent_trace"][-1]
    mode = "ReAct (LLM)" if trace.reasoning_steps > 0 else "Rule-based"
    print(f"\n  Mode: {mode} | Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_step3(state: FactCheckState) -> None:
    """Print evidence retriever output."""
    _hr("STEP 3: Evidence Retriever (ReAct agent)")
    print("What it does:")
    print("  Executes the retrieval plan — calls PubMed, Semantic Scholar,")
    print("  Cochrane, ClinicalTrials.gov, DrugBank as planned.")
    print("  Re-ranks with cross-encoder, deduplicates, links to sub-claims.")
    _hr()

    evidence = state.get("evidence", [])
    sub_claims = state.get("sub_claims", [])

    for sc in sub_claims:
        sc_evidence = [e for e in evidence if e.id in sc.evidence]
        print(f"  [{sc.id}] \"{sc.text}\"")
        print(f"    Evidence items: {len(sc_evidence)}")
        if sc_evidence:
            by_source: dict[str, int] = {}
            for ev in sc_evidence:
                by_source[ev.source] = by_source.get(ev.source, 0) + 1
            source_summary = ", ".join(f"{s}: {c}" for s, c in sorted(by_source.items()))
            print(f"    Sources: {source_summary}")
        print()

    if evidence:
        sorted_ev = sorted(evidence, key=lambda e: e.quality_score, reverse=True)
        top_n = min(5, len(sorted_ev))
        print(f"  Top {top_n} evidence items (by quality score):")
        print()
        for i, ev in enumerate(sorted_ev[:top_n], 1):
            print(f"    {i}. [{ev.source}] {ev.title[:70]}")
            print(f"       Study type: {ev.study_type or 'unknown'} | "
                  f"Score: {ev.quality_score:.3f} | "
                  f"PMID: {ev.pmid or '—'}")
            content_preview = ev.content[:120].replace("\n", " ")
            print(f"       {content_preview}...")
            if ev.url:
                print(f"       URL: {ev.url}")
            print()

    print(f"  Total evidence: {len(evidence)}")
    trace = state["agent_trace"][-1]
    mode = "ReAct (LLM)" if trace.reasoning_steps > 0 else "Rule-based"
    print(f"  Mode: {mode} | Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_step4(state: FactCheckState) -> None:
    """Print evidence grader output."""
    _hr("STEP 4: Evidence Grader (ReAct agent)")
    print("What it does:")
    print("  Evaluates each evidence item for study type, methodology,")
    print("  relevance to sub-claims, and applies the GRADE framework.")
    print("  Produces per-evidence quality scores and per-subclaim summaries.")
    _hr()

    eq = state.get("evidence_quality", {})
    per_evidence = eq.get("per_evidence", {})
    per_subclaim = eq.get("per_subclaim", {})

    if per_evidence:
        sorted_ev = sorted(
            per_evidence.items(),
            key=lambda x: x[1].get("evidence_strength", 0),
            reverse=True,
        )
        top_n = min(5, len(sorted_ev))
        print(f"  Top {top_n} evidence by strength:")
        print()
        for i, (ev_id, info) in enumerate(sorted_ev[:top_n], 1):
            print(f"    {i}. [{ev_id}]")
            print(f"       Study type: {info.get('study_type', '?')} "
                  f"(weight: {info.get('hierarchy_weight', 0):.1f})")
            print(f"       Methodology: {info.get('methodology_score', 0):.2f}")
            print(f"       Evidence strength: {info.get('evidence_strength', 0):.3f}")
            for sc_id, rel in info.get("relevance", {}).items():
                print(f"       → {sc_id}: {rel.get('direction', '?')} "
                      f"(score: {rel.get('score', 0):.2f})")
                if rel.get("key_finding"):
                    finding = rel["key_finding"][:80]
                    print(f"         Finding: {finding}")
            print()

    if per_subclaim:
        print("  Per-subclaim evidence summary:")
        for sc_id, info in per_subclaim.items():
            sc = next((s for s in state.get("sub_claims", []) if s.id == sc_id), None)
            label = f"\"{sc.text}\"" if sc else sc_id
            print(f"\n    [{sc_id}] {label}")
            print(f"      Evidence count: {info.get('evidence_count', 0)}")
            print(f"      Avg strength: {info.get('avg_strength', 0):.3f}")
            ds = info.get("direction_summary", {})
            print(f"      Directions: supports={ds.get('supports', 0)}, "
                  f"opposes={ds.get('opposes', 0)}, "
                  f"neutral={ds.get('neutral', 0)}")
            top_ids = info.get("top_evidence_ids", [])
            if top_ids:
                print(f"      Top evidence: {', '.join(top_ids[:3])}")

    print()
    print(f"  Total graded: {len(per_evidence)}")
    trace = state["agent_trace"][-1]
    mode = "ReAct (LLM)" if trace.reasoning_steps > 0 else "Rule-based"
    print(f"  Mode: {mode} | Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_step5(state: FactCheckState) -> None:
    """Print verdict agent output."""
    _hr("STEP 5: Verdict Agent (ReAct agent)")
    print("What it does:")
    print("  Weighs evidence for/against each sub-claim, reconciles")
    print("  conflicts, assigns per-sub-claim verdicts, and synthesizes")
    print("  an overall 9-level verdict with confidence and explanation.")
    _hr()

    # Per-subclaim verdicts
    sub_claims = state.get("sub_claims", [])
    print(f"  Per-subclaim verdicts ({len(sub_claims)}):")
    for sc in sub_claims:
        verdict = sc.verdict or "pending"
        confidence = sc.confidence
        print(f"    [{sc.id}] \"{sc.text}\"")
        print(f"      Verdict: {verdict} (confidence: {confidence:.2f})")
    print()

    # Overall verdict
    print(f"  Overall verdict: {state.get('verdict', '?')}")
    print(f"  Confidence: {state.get('confidence', 0.0):.2f}")
    explanation = state.get("explanation", "")
    if explanation:
        print(f"  Explanation:")
        print(_wrap(explanation))

    print()
    trace = state["agent_trace"][-1]
    mode = "ReAct (LLM)" if trace.reasoning_steps > 0 else "Rule-based"
    print(f"  Mode: {mode} | Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_step6(state: FactCheckState) -> None:
    """Print safety checker output."""
    _hr("STEP 6: Safety Checker (function node)")
    print("What it does:")
    print("  Scans claim for dangerous health advice patterns across")
    print("  6 categories. Flags critical safety concerns and marks")
    print("  claims as dangerous if they could cause harm.")
    _hr()

    safety_flags = state.get("safety_flags", [])
    is_dangerous = state.get("is_dangerous", False)

    if safety_flags:
        print(f"  Safety flags ({len(safety_flags)}):")
        from src.functions.safety_checker import SAFETY_CATEGORIES
        for flag in safety_flags:
            desc = SAFETY_CATEGORIES.get(flag, flag)
            print(f"    - {flag}: {desc}")
    else:
        print("  No safety flags triggered.")

    print(f"\n  Is dangerous: {is_dangerous}")

    print()
    trace = state["agent_trace"][-1]
    print(f"  Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_summary(state: FactCheckState) -> None:
    """Print pipeline summary."""
    _hr("PIPELINE SUMMARY")
    print(f"  Claim: \"{state['claim']}\"")
    print(f"  Sub-claims: {len(state.get('sub_claims', []))}")
    print(f"  Evidence items: {len(state.get('evidence', []))}")
    eq = state.get("evidence_quality", {})
    print(f"  Graded evidence: {len(eq.get('per_evidence', {}))}")
    print()
    print(f"  VERDICT: {state.get('verdict', '?')}")
    print(f"  Confidence: {state.get('confidence', 0.0):.2f}")
    print(f"  Dangerous: {state.get('is_dangerous', False)}")
    safety_flags = state.get("safety_flags", [])
    if safety_flags:
        print(f"  Safety flags: {', '.join(safety_flags)}")
    print()
    print(f"  Total cost: ${state.get('total_cost_usd', 0):.6f}")
    print(f"  Total duration: {state.get('total_duration_seconds', 0):.2f}s")
    print()

    print("  Node trace:")
    for trace in state.get("agent_trace", []):
        print(f"    {trace.agent} ({trace.node_type}) — "
              f"{trace.duration_seconds}s, ${trace.cost_usd:.6f}")
    print()


async def main(claim: str) -> None:
    print(f"\n  Claim: \"{claim}\"")

    # Build initial state
    state: FactCheckState = {
        "claim": claim,
        "pico": None,
        "sub_claims": [],
        "entities": {},
        "retrieval_plan": {},
        "evidence": [],
        "extracted_figures": [],
        "evidence_quality": {},
        "verdict": "",
        "confidence": 0.0,
        "explanation": "",
        "safety_flags": [],
        "is_dangerous": False,
        "agent_trace": [],
        "total_cost_usd": 0.0,
        "total_duration_seconds": 0.0,
    }

    # Step 1: Decomposer
    print("\n  Running decomposer...", flush=True)
    state = await run_decomposer(state)
    print_step1(state)

    # Step 2: Retrieval Planner
    print("\n  Running retrieval planner...", flush=True)
    state = await run_retrieval_planner(state)
    print_step2(state)

    # Step 3: Evidence Retriever
    print("\n  Running evidence retriever...", flush=True)
    state = await run_evidence_retriever(state)
    print_step3(state)

    # Step 4: Evidence Grader
    print("\n  Running evidence grader...", flush=True)
    state = await run_evidence_grader(state)
    print_step4(state)

    # Step 5: Verdict Agent
    print("\n  Running verdict agent...", flush=True)
    state = await run_verdict_agent(state)
    print_step5(state)

    # Step 6: Safety Checker
    print("\n  Running safety checker...", flush=True)
    state = await run_safety_checker(state)
    print_step6(state)

    # Summary
    print_summary(state)


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLAIM
    asyncio.run(main(claim))
