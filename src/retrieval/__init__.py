"""Retrieval clients for biomedical literature and drug information."""

from src.retrieval.pubmed_client import PubMedArticle, search_and_fetch as pubmed_search
from src.retrieval.semantic_scholar import S2Paper, search as s2_search
from src.retrieval.cochrane_client import CochraneReview, search_cochrane
from src.retrieval.clinical_trials import ClinicalTrial, search as ct_search
from src.retrieval.drugbank_client import DrugInfo, DrugInteraction, search_drug_label, get_interactions

__all__ = [
    "PubMedArticle",
    "pubmed_search",
    "S2Paper",
    "s2_search",
    "CochraneReview",
    "search_cochrane",
    "ClinicalTrial",
    "ct_search",
    "DrugInfo",
    "DrugInteraction",
    "search_drug_label",
    "get_interactions",
]
