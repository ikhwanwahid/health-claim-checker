#!/bin/bash
# Health Claims Fact-Checker - Project Setup Script
# Run this first to create the directory structure

set -e

echo "🏥 Setting up Health Claims Fact-Checker project..."

# Create main directories
mkdir -p data/{benchmarks/{scifact,pubhealth,healthver},claims,guidelines/{who,nih,moh_singapore},figures/{forest_plots,kaplan_meier}}
mkdir -p src/{agents,graph,retrieval,medical_nlp,evaluation}
mkdir -p app
mkdir -p notebooks
mkdir -p scripts
mkdir -p tests/{test_agents,test_retrieval,test_evaluation}
mkdir -p docs

# Create __init__.py files
touch src/__init__.py
touch src/agents/__init__.py
touch src/graph/__init__.py
touch src/retrieval/__init__.py
touch src/medical_nlp/__init__.py
touch src/evaluation/__init__.py
touch tests/__init__.py
touch tests/test_agents/__init__.py
touch tests/test_retrieval/__init__.py
touch tests/test_evaluation/__init__.py

# Create placeholder files
touch data/claims/curated_claims.json
touch data/claims/perturbed_claims.json
touch data/claims/ground_truth.json
touch data/figures/ground_truth.json

# Create .env.example
cat > .env.example << 'EOF'
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional (higher rate limits)
NCBI_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=

# Optional (for observability)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project
.env
*.log
data/benchmarks/*
!data/benchmarks/.gitkeep
data/guidelines/*
!data/guidelines/.gitkeep

# Notebooks
.ipynb_checkpoints/

# Build
dist/
build/
*.egg-info/
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core
langchain>=0.1.0
langgraph>=0.0.20
anthropic>=0.18.0
openai>=1.0.0
pydantic>=2.0.0

# Retrieval
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
biopython>=1.80

# Medical NLP
scispacy>=0.5.0
# Run separately: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz

# Data
pandas>=2.0.0
pdfplumber>=0.9.0
requests>=2.28.0

# UI
streamlit>=1.28.0

# Observability
langfuse>=2.0.0

# Dev
pytest>=7.0.0
python-dotenv>=1.0.0
ipykernel>=6.0.0
EOF

# Create basic config.py
cat > src/config.py << 'EOF'
"""Configuration and settings for Health Claims Fact-Checker."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
GUIDELINES_DIR = DATA_DIR / "guidelines"

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# Model configs
CLAUDE_MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"

# Evidence hierarchy weights
EVIDENCE_WEIGHTS = {
    "guideline": 1.0,
    "systematic_review": 0.9,
    "meta_analysis": 0.9,
    "rct": 0.8,
    "cohort": 0.6,
    "case_control": 0.5,
    "case_report": 0.3,
    "in_vitro": 0.2,
    "expert_opinion": 0.1,
}

# Verdict types
VERDICTS = [
    "SUPPORTED",
    "SUPPORTED_WITH_CAVEATS",
    "OVERSTATED",
    "MISLEADING",
    "PRELIMINARY",
    "OUTDATED",
    "NOT_SUPPORTED",
    "REFUTED",
    "DANGEROUS",
]
EOF

# Create basic state.py
cat > src/graph/state.py << 'EOF'
"""LangGraph state definitions for the fact-checking pipeline."""

from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel


class PICO(BaseModel):
    """PICO framework extraction."""
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparison: Optional[str] = None
    outcome: Optional[str] = None


class SubClaim(BaseModel):
    """A decomposed sub-claim."""
    id: str
    text: str
    pico: Optional[PICO] = None
    verdict: Optional[str] = None
    evidence: List[str] = []
    confidence: float = 0.0


class Evidence(BaseModel):
    """A piece of retrieved evidence."""
    id: str
    source: str  # pubmed, cochrane, guideline, etc.
    retrieval_method: str  # api, cross_encoder, deep_search, vlm
    title: str
    content: str
    url: Optional[str] = None
    study_type: Optional[str] = None
    quality_score: float = 0.0
    pmid: Optional[str] = None


class AgentTrace(BaseModel):
    """Trace of a single agent's execution."""
    agent: str
    duration_seconds: float
    cost_usd: float
    input_summary: str
    output_summary: str
    success: bool


class FactCheckState(TypedDict):
    """Main state passed through the LangGraph workflow."""
    # Input
    claim: str
    
    # Decomposition
    pico: Optional[PICO]
    sub_claims: List[SubClaim]
    entities: dict
    
    # Retrieval planning
    retrieval_plan: dict
    
    # Evidence
    evidence: List[Evidence]
    
    # VLM
    extracted_figures: List[dict]
    
    # Grading
    evidence_quality: dict
    
    # Verdict
    verdict: str
    confidence: float
    explanation: str
    
    # Safety
    safety_flags: List[str]
    is_dangerous: bool
    
    # Tracing
    agent_trace: List[AgentTrace]
    total_cost_usd: float
    total_duration_seconds: float
EOF

# Create placeholder agent files
for agent in decomposer retrieval_planner evidence_retriever vlm_extractor evidence_grader verdict_agent safety_checker; do
    cat > src/agents/${agent}.py << EOF
"""${agent} agent implementation."""

from src.graph.state import FactCheckState


async def run_${agent}(state: FactCheckState) -> FactCheckState:
    """Run the ${agent} agent.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state
    """
    # TODO: Implement
    raise NotImplementedError("${agent} not yet implemented")
EOF
done

# Create basic workflow.py
cat > src/graph/workflow.py << 'EOF'
"""Main LangGraph workflow for health claim verification."""

from langgraph.graph import StateGraph, END
from src.graph.state import FactCheckState
from src.functions import decomposer, safety_checker
from src.agents import (
    retrieval_planner,
    evidence_retriever,
    vlm_extractor,
    evidence_grader,
    verdict_agent,
)


def create_workflow() -> StateGraph:
    """Create the fact-checking workflow graph."""
    
    workflow = StateGraph(FactCheckState)
    
    # Add nodes
    workflow.add_node("decomposer", decomposer.run_decomposer)
    workflow.add_node("retrieval_planner", retrieval_planner.run_retrieval_planner)
    workflow.add_node("evidence_retriever", evidence_retriever.run_evidence_retriever)
    workflow.add_node("vlm_extractor", vlm_extractor.run_vlm_extractor)
    workflow.add_node("evidence_grader", evidence_grader.run_evidence_grader)
    workflow.add_node("verdict_agent", verdict_agent.run_verdict_agent)
    workflow.add_node("safety_checker", safety_checker.run_safety_checker)
    
    # Define edges
    workflow.set_entry_point("decomposer")
    workflow.add_edge("decomposer", "retrieval_planner")
    workflow.add_edge("retrieval_planner", "evidence_retriever")
    workflow.add_edge("evidence_retriever", "vlm_extractor")
    workflow.add_edge("vlm_extractor", "evidence_grader")
    workflow.add_edge("evidence_grader", "verdict_agent")
    workflow.add_edge("verdict_agent", "safety_checker")
    workflow.add_edge("safety_checker", END)
    
    return workflow.compile()


async def verify_claim(claim: str) -> FactCheckState:
    """Verify a health claim.
    
    Args:
        claim: The health claim to verify
        
    Returns:
        Final state with verdict and evidence
    """
    workflow = create_workflow()
    
    initial_state: FactCheckState = {
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
    
    result = await workflow.ainvoke(initial_state)
    return result
EOF

# Create basic Streamlit app
cat > app/streamlit_app.py << 'EOF'
"""Health Claims Fact-Checker - Streamlit UI."""

import streamlit as st

st.set_page_config(
    page_title="Health Claims Fact-Checker",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Health Claims Fact-Checker")
st.markdown("Verify health claims against peer-reviewed research, clinical trials, and medical guidelines.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    system_variant = st.selectbox(
        "System Variant",
        ["S5: Full Pipeline", "S4: Multi-method", "B3: API + Vector DB", "B2: API-only", "B1: No retrieval"],
    )
    
    st.divider()
    
    st.subheader("Retrieval Methods")
    use_pubmed = st.checkbox("PubMed API", value=True)
    use_scholar = st.checkbox("Semantic Scholar", value=True)
    use_cochrane = st.checkbox("Cochrane", value=True)
    use_trials = st.checkbox("ClinicalTrials.gov", value=True)
    use_guidelines = st.checkbox("Guideline Store", value=True)
    use_deep_search = st.checkbox("Full-text Deep Search", value=True)
    use_vlm = st.checkbox("VLM Figure Extraction", value=True)
    
    st.divider()
    
    show_trace = st.checkbox("Show Agent Trace", value=True)
    show_cost = st.checkbox("Show Cost Breakdown", value=True)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    claim = st.text_area(
        "Enter a health claim to verify:",
        placeholder="e.g., Intermittent fasting reverses Type 2 diabetes",
        height=100,
    )

with col2:
    st.markdown("**Try an example:**")
    examples = [
        "Intermittent fasting reverses Type 2 diabetes",
        "Vitamin D prevents COVID-19",
        "Vaccines cause autism",
        "Turmeric cures cancer",
        "Aspirin is a blood thinner",
    ]
    for ex in examples:
        if st.button(ex[:30] + "...", key=ex):
            claim = ex

if st.button("🔍 Verify Claim", type="primary"):
    if not claim:
        st.warning("Please enter a claim to verify.")
    else:
        with st.spinner("Verifying claim..."):
            # TODO: Call the actual pipeline
            st.info("Pipeline not yet implemented. This is a placeholder UI.")
            
            # Placeholder verdict
            st.success("## ⚠️ OVERSTATED")
            st.metric("Confidence", "78%")
            st.markdown("""
            **Explanation:** While intermittent fasting shows promise for glycemic control,
            calling it a "reversal" overstates the evidence. Most studies show improvement,
            not remission. Long-term evidence is limited.
            """)

# Tabs for details
tab1, tab2, tab3, tab4 = st.tabs(["Agent Trace", "Evidence", "Figures", "Retrieval Methods"])

with tab1:
    st.markdown("### Agent Execution Trace")
    st.info("Agent trace will appear here after verification.")

with tab2:
    st.markdown("### Retrieved Evidence")
    st.info("Evidence will appear here after verification.")

with tab3:
    st.markdown("### Extracted Figures")
    st.info("VLM-extracted figures will appear here after verification.")

with tab4:
    st.markdown("### Retrieval Methods Used")
    st.info("Retrieval method breakdown will appear here after verification.")
EOF

# Create download benchmarks script
cat > scripts/download_benchmarks.py << 'EOF'
"""Download benchmark datasets for evaluation."""

import os
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks"


def download_scifact():
    """Download SciFact dataset."""
    print("Downloading SciFact...")
    # TODO: Implement actual download
    # https://github.com/allenai/scifact
    (DATA_DIR / "scifact" / ".gitkeep").touch()
    print("SciFact: placeholder created (implement actual download)")


def download_pubhealth():
    """Download PUBHEALTH dataset."""
    print("Downloading PUBHEALTH...")
    # TODO: Implement actual download
    # https://github.com/neemakot/Health-Fact-Checking
    (DATA_DIR / "pubhealth" / ".gitkeep").touch()
    print("PUBHEALTH: placeholder created (implement actual download)")


def download_healthver():
    """Download HealthVer dataset."""
    print("Downloading HealthVer...")
    # TODO: Implement actual download
    (DATA_DIR / "healthver" / ".gitkeep").touch()
    print("HealthVer: placeholder created (implement actual download)")


if __name__ == "__main__":
    download_scifact()
    download_pubhealth()
    download_healthver()
    print("\n✅ Benchmark download complete!")
EOF

echo ""
echo "✅ Project structure created!"
echo ""
echo "Next steps:"
echo "  1. cd health-claim-checker"
echo "  2. python -m venv venv"
echo "  3. source venv/bin/activate  (or venv\\Scripts\\activate on Windows)"
echo "  4. pip install -r requirements.txt"
echo "  5. cp .env.example .env  (and add your API keys)"
echo "  6. streamlit run app/streamlit_app.py"
echo ""
echo "For Claude Code:"
echo "  The CLAUDE.md file is ready. Claude Code will automatically read it."
echo ""
