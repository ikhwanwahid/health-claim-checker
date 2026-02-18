"""End-to-end demo: Decomposer → Retrieval Planner → Evidence Retriever.

Runs the first 3 pipeline nodes on a claim and prints detailed output
at each step so you can see what each node does.

Usage:
    uv run python scripts/e2e_retriever_demo.py
    uv run python scripts/e2e_retriever_demo.py "Your custom claim here"
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

from src.functions.decomposer import run_decomposer
from src.models import FactCheckState
from systems.s4_langgraph.agents.retrieval_planner import run_retrieval_planner
from systems.s4_langgraph.agents.evidence_retriever import run_evidence_retriever


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

    # Entities
    entities = state.get("entities", {})
    print("  Extracted entities:")
    for etype, elist in entities.items():
        if elist:
            print(f"    {etype}: {', '.join(elist)}")
    if not any(v for v in entities.values()):
        print("    (none detected)")

    # PICO
    pico = state.get("pico")
    print(f"\n  PICO extraction:")
    if pico:
        print(f"    Population:   {pico.population or '—'}")
        print(f"    Intervention: {pico.intervention or '—'}")
        print(f"    Comparison:   {pico.comparison or '—'}")
        print(f"    Outcome:      {pico.outcome or '—'}")
    else:
        print("    (no PICO)")

    # Sub-claims
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

    # Summary per sub-claim
    for sc in sub_claims:
        sc_evidence = [e for e in evidence if e.id in sc.evidence]
        print(f"  [{sc.id}] \"{sc.text}\"")
        print(f"    Evidence items: {len(sc_evidence)}")
        if sc_evidence:
            # Group by source
            by_source: dict[str, int] = {}
            for ev in sc_evidence:
                by_source[ev.source] = by_source.get(ev.source, 0) + 1
            source_summary = ", ".join(f"{s}: {c}" for s, c in sorted(by_source.items()))
            print(f"    Sources: {source_summary}")
        print()

    # Show top evidence items
    if evidence:
        # Sort by quality_score descending
        sorted_ev = sorted(evidence, key=lambda e: e.quality_score, reverse=True)
        top_n = min(5, len(sorted_ev))
        print(f"  Top {top_n} evidence items (by quality score):")
        print()
        for i, ev in enumerate(sorted_ev[:top_n], 1):
            print(f"    {i}. [{ev.source}] {ev.title[:70]}")
            print(f"       Study type: {ev.study_type or 'unknown'} | "
                  f"Score: {ev.quality_score:.3f} | "
                  f"PMID: {ev.pmid or '—'}")
            # Truncated abstract
            content_preview = ev.content[:120].replace("\n", " ")
            print(f"       {content_preview}...")
            if ev.url:
                print(f"       URL: {ev.url}")
            print()

    print(f"  Total evidence: {len(evidence)}")
    trace = state["agent_trace"][-1]
    mode = "ReAct (LLM)" if trace.reasoning_steps > 0 else "Rule-based"
    print(f"  Mode: {mode} | Duration: {trace.duration_seconds}s | Cost: ${trace.cost_usd:.6f}")


def print_summary(state: FactCheckState) -> None:
    """Print pipeline summary."""
    _hr("PIPELINE SUMMARY")
    print(f"  Claim: \"{state['claim']}\"")
    print(f"  Sub-claims: {len(state.get('sub_claims', []))}")
    print(f"  Evidence items: {len(state.get('evidence', []))}")
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

    # Summary
    print_summary(state)


if __name__ == "__main__":
    claim = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CLAIM
    asyncio.run(main(claim))
