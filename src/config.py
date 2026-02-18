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
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

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
