"""Medical Named Entity Recognition using scispaCy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import spacy

logger = logging.getLogger(__name__)

# Lazy-loaded singleton
_nlp: Optional[spacy.language.Language] = None


def _get_nlp() -> spacy.language.Language:
    """Load the best available spaCy model once and cache it.

    Tries scispaCy models first (best for biomedical text), then falls back
    to standard spaCy models.
    """
    global _nlp
    if _nlp is None:
        # Try models in order of preference for biomedical NER
        models = [
            "en_core_sci_lg",   # scispaCy large (best)
            "en_core_sci_sm",   # scispaCy small
            "en_core_web_sm",   # standard spaCy (fallback)
        ]
        for model_name in models:
            try:
                _nlp = spacy.load(model_name)
                logger.info("Loaded NLP model: %s", model_name)
                break
            except OSError:
                continue

        if _nlp is None:
            raise RuntimeError(
                "No spaCy model found. Install one with: "
                "python -m spacy download en_core_web_sm"
            )
    return _nlp


@dataclass
class MedicalEntities:
    """Extracted medical entities from text."""
    drugs: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    organisms: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    anatomical: list[str] = field(default_factory=list)
    raw_entities: list[dict] = field(default_factory=list)

    def all_entities(self) -> list[str]:
        """Return all entities as a flat list."""
        return (
            self.drugs
            + self.conditions
            + self.genes
            + self.organisms
            + self.procedures
            + self.anatomical
        )

    def has_drugs(self) -> bool:
        return len(self.drugs) > 0


# Heuristic keyword sets for classifying scispaCy entities
_DRUG_HINTS = {
    "mg", "dose", "tablet", "capsule", "drug", "medication",
    "inhibitor", "blocker", "agonist", "antagonist", "statin",
    "aspirin", "ibuprofen", "metformin", "insulin",
}
_CONDITION_HINTS = {
    "disease", "disorder", "syndrome", "cancer", "tumor", "infection",
    "diabetes", "hypertension", "obesity", "depression", "asthma",
    "deficiency", "failure", "inflammation",
}
_GENE_HINTS = {
    "gene", "mutation", "variant", "allele", "polymorphism",
    "expression", "receptor", "kinase", "transcription",
}
_PROCEDURE_HINTS = {
    "surgery", "therapy", "treatment", "transplant", "procedure",
    "intervention", "screening", "biopsy", "imaging",
}


def _classify_entity(text: str, label: str) -> str:
    """Classify an entity into a category using label + heuristics."""
    text_lower = text.lower()

    # Use spaCy label if available and informative
    label_upper = label.upper()
    if label_upper in ("CHEMICAL", "SIMPLE_CHEMICAL"):
        return "drug"
    if label_upper in ("DISEASE", "DISORDER"):
        return "condition"
    if label_upper in ("GENE_OR_GENE_PRODUCT", "GENE"):
        return "gene"
    if label_upper in ("ORGANISM", "TAXON"):
        return "organism"
    if label_upper in ("ORGAN", "TISSUE", "CELL", "ANATOMY"):
        return "anatomical"

    # Fallback: keyword heuristics
    words = set(text_lower.split())
    if words & _DRUG_HINTS:
        return "drug"
    if words & _CONDITION_HINTS:
        return "condition"
    if words & _GENE_HINTS:
        return "gene"
    if words & _PROCEDURE_HINTS:
        return "procedure"

    # Check if any hint word appears as a substring
    for hint in _CONDITION_HINTS:
        if hint in text_lower:
            return "condition"
    for hint in _DRUG_HINTS:
        if hint in text_lower:
            return "drug"

    return "condition"  # Default for medical text


def extract_entities(text: str) -> MedicalEntities:
    """Extract medical entities from text using scispaCy.

    Args:
        text: Input text (e.g., a health claim).

    Returns:
        MedicalEntities with categorized entities.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    entities = MedicalEntities()
    seen = set()

    for ent in doc.ents:
        ent_text = ent.text.strip()
        if not ent_text or ent_text.lower() in seen:
            continue
        seen.add(ent_text.lower())

        category = _classify_entity(ent_text, ent.label_)
        raw = {"text": ent_text, "label": ent.label_, "category": category,
               "start": ent.start_char, "end": ent.end_char}
        entities.raw_entities.append(raw)

        if category == "drug":
            entities.drugs.append(ent_text)
        elif category == "condition":
            entities.conditions.append(ent_text)
        elif category == "gene":
            entities.genes.append(ent_text)
        elif category == "organism":
            entities.organisms.append(ent_text)
        elif category == "procedure":
            entities.procedures.append(ent_text)
        elif category == "anatomical":
            entities.anatomical.append(ent_text)

    return entities
