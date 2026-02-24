"""Tests for the unified benchmark dataset loader."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.evaluation.dataset_loader import (
    BenchmarkClaim,
    filter_health_claims,
    load_covidfact,
    load_dataset,
    load_healthver,
    load_pubhealth,
    load_scifact,
)


# ---------------------------------------------------------------------------
# Fixtures — small mock data files
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_data_dir(tmp_path, monkeypatch):
    """Create a temporary benchmarks directory with small fixture files."""
    benchmarks = tmp_path / "data" / "benchmarks"

    # --- SciFact fixtures ---
    scifact_dir = benchmarks / "scifact"
    scifact_dir.mkdir(parents=True)

    scifact_claims = [
        {"id": 1, "claim": "Vitamin D prevents respiratory infections", "evidence": {"12345": [{"label": "SUPPORT", "sentences": [0, 1]}]}, "cited_doc_ids": [12345]},
        {"id": 2, "claim": "Hydroxychloroquine cures COVID-19", "evidence": {"67890": [{"label": "CONTRADICT", "sentences": [2]}]}, "cited_doc_ids": [67890]},
        {"id": 3, "claim": "Exercise improves mental health", "evidence": {}, "cited_doc_ids": []},
    ]
    (scifact_dir / "claims_dev.jsonl").write_text(
        "\n".join(json.dumps(c) for c in scifact_claims), encoding="utf-8"
    )
    (scifact_dir / "claims_train.jsonl").write_text(
        json.dumps(scifact_claims[0]), encoding="utf-8"
    )
    (scifact_dir / "claims_test.jsonl").write_text(
        json.dumps(scifact_claims[1]), encoding="utf-8"
    )

    # --- PUBHEALTH fixtures ---
    pubhealth_dir = benchmarks / "pubhealth"
    pubhealth_dir.mkdir(parents=True)

    pubhealth_tsv = textwrap.dedent("""\
        claim_id\tclaim\tlabel\texplanation\tmain_text\tsources
        1\tMMR vaccine causes autism\tfalse\tExtensive research shows no link\tMultiple studies...\thttps://example.com
        2\tVitamin C cures the common cold\tmixture\tMay reduce duration slightly\tSome evidence...\thttps://example.com
        3\tSmoking causes lung cancer\ttrue\tOverwhelming evidence\tDecades of research...\thttps://example.com
        4\tHomeopathy treats cancer\tunproven\tNo scientific evidence\tNo clinical trials...\thttps://example.com
    """)
    (pubhealth_dir / "test.tsv").write_text(pubhealth_tsv, encoding="utf-8")
    (pubhealth_dir / "train.tsv").write_text(pubhealth_tsv, encoding="utf-8")
    (pubhealth_dir / "dev.tsv").write_text(pubhealth_tsv, encoding="utf-8")

    # --- HealthVer fixtures ---
    healthver_dir = benchmarks / "healthver"
    healthver_dir.mkdir(parents=True)

    healthver_csv = textwrap.dedent("""\
        claim_id,claim,label,evidence
        1,COVID-19 vaccines are effective,SUPPORT,Clinical trials show 95% efficacy
        2,5G causes coronavirus,REFUTE,There is no evidence linking 5G to viruses
        3,Masks reduce transmission,NEUTRAL,Some studies show mixed results
    """)
    (healthver_dir / "healthver_dev.csv").write_text(healthver_csv, encoding="utf-8")
    (healthver_dir / "healthver_train.csv").write_text(healthver_csv, encoding="utf-8")
    (healthver_dir / "healthver_test.csv").write_text(healthver_csv, encoding="utf-8")

    # --- COVID-Fact fixtures ---
    covidfact_dir = benchmarks / "covidfact"
    covidfact_dir.mkdir(parents=True)

    covidfact_entries = [
        {"claim": "COVID-19 vaccine contains microchips", "label": "REFUTED", "evidence": ["No evidence of microchips in vaccines"], "gold_source": "https://example.com"},
        {"claim": "Masks reduce COVID-19 spread", "label": "SUPPORTED", "evidence": ["Multiple studies confirm", "masks reduce transmission"], "gold_source": "https://example.com"},
    ]
    (covidfact_dir / "COVIDFACT_dataset.jsonl").write_text(
        "\n".join(json.dumps(e) for e in covidfact_entries), encoding="utf-8"
    )

    # Patch DATA_DIR in the loader module
    import src.evaluation.dataset_loader as loader_module
    monkeypatch.setattr(loader_module, "DATA_DIR", benchmarks)

    return benchmarks


# ---------------------------------------------------------------------------
# BenchmarkClaim structure
# ---------------------------------------------------------------------------

class TestBenchmarkClaim:
    def test_fields(self):
        claim = BenchmarkClaim(
            id="1", claim="Test claim", label="true",
            dataset="pubhealth", evidence_text="evidence", split="test",
        )
        assert claim.id == "1"
        assert claim.claim == "Test claim"
        assert claim.label == "true"
        assert claim.dataset == "pubhealth"
        assert claim.evidence_text == "evidence"
        assert claim.split == "test"

    def test_equality(self):
        a = BenchmarkClaim("1", "claim", "true", "ds", "ev", "test")
        b = BenchmarkClaim("1", "claim", "true", "ds", "ev", "test")
        assert a == b


# ---------------------------------------------------------------------------
# SciFact loader
# ---------------------------------------------------------------------------

class TestLoadScifact:
    def test_load_dev(self, mock_data_dir):
        claims = load_scifact(split="dev")
        assert len(claims) == 3

    def test_label_derivation_support(self, mock_data_dir):
        claims = load_scifact(split="dev")
        claim_1 = next(c for c in claims if c.id == "1")
        assert claim_1.label == "SUPPORTS"

    def test_label_derivation_contradict(self, mock_data_dir):
        claims = load_scifact(split="dev")
        claim_2 = next(c for c in claims if c.id == "2")
        assert claim_2.label == "REFUTES"

    def test_label_derivation_nei(self, mock_data_dir):
        claims = load_scifact(split="dev")
        claim_3 = next(c for c in claims if c.id == "3")
        assert claim_3.label == "NEI"

    def test_dataset_field(self, mock_data_dir):
        claims = load_scifact(split="dev")
        assert all(c.dataset == "scifact" for c in claims)

    def test_split_field(self, mock_data_dir):
        claims = load_scifact(split="train")
        assert all(c.split == "train" for c in claims)

    def test_invalid_split(self, mock_data_dir):
        with pytest.raises(ValueError, match="Invalid split"):
            load_scifact(split="nonexistent")

    def test_missing_file(self):
        """Test graceful error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="SciFact file not found"):
            # Use a split that points to a nonexistent path (default DATA_DIR)
            load_scifact(split="dev")


# ---------------------------------------------------------------------------
# PUBHEALTH loader
# ---------------------------------------------------------------------------

class TestLoadPubhealth:
    def test_load_test(self, mock_data_dir):
        claims = load_pubhealth(split="test")
        assert len(claims) == 4

    def test_labels(self, mock_data_dir):
        claims = load_pubhealth(split="test")
        labels = {c.label for c in claims}
        assert labels == {"true", "false", "mixture", "unproven"}

    def test_dataset_field(self, mock_data_dir):
        claims = load_pubhealth(split="test")
        assert all(c.dataset == "pubhealth" for c in claims)

    def test_evidence_text(self, mock_data_dir):
        claims = load_pubhealth(split="test")
        # Should have explanation as evidence_text (preferred over main_text)
        claim_1 = next(c for c in claims if c.id == "1")
        assert "no link" in claim_1.evidence_text.lower()

    def test_invalid_split(self, mock_data_dir):
        with pytest.raises(ValueError, match="Invalid split"):
            load_pubhealth(split="nonexistent")

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="PUBHEALTH file not found"):
            load_pubhealth(split="test")


# ---------------------------------------------------------------------------
# HealthVer loader
# ---------------------------------------------------------------------------

class TestLoadHealthver:
    def test_load_dev(self, mock_data_dir):
        claims = load_healthver(split="dev")
        assert len(claims) == 3

    def test_label_normalization(self, mock_data_dir):
        claims = load_healthver(split="dev")
        labels = {c.label for c in claims}
        assert labels == {"SUPPORTS", "REFUTES", "NEI"}

    def test_dataset_field(self, mock_data_dir):
        claims = load_healthver(split="dev")
        assert all(c.dataset == "healthver" for c in claims)

    def test_evidence_text(self, mock_data_dir):
        claims = load_healthver(split="dev")
        claim_1 = next(c for c in claims if c.id == "1")
        assert "95% efficacy" in claim_1.evidence_text

    def test_invalid_split(self, mock_data_dir):
        with pytest.raises(ValueError, match="Invalid split"):
            load_healthver(split="nonexistent")

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="HealthVer file not found"):
            load_healthver(split="dev")


# ---------------------------------------------------------------------------
# COVID-Fact loader
# ---------------------------------------------------------------------------

class TestLoadCovidfact:
    def test_load(self, mock_data_dir):
        claims = load_covidfact()
        assert len(claims) == 2

    def test_label_mapping(self, mock_data_dir):
        claims = load_covidfact()
        # REFUTED → "false", SUPPORTED → "true"
        refuted = next(c for c in claims if "microchip" in c.claim.lower())
        supported = next(c for c in claims if "mask" in c.claim.lower())
        assert refuted.label == "false"
        assert supported.label == "true"

    def test_evidence_concatenation(self, mock_data_dir):
        claims = load_covidfact()
        supported = next(c for c in claims if "mask" in c.claim.lower())
        assert "Multiple studies confirm" in supported.evidence_text
        assert "masks reduce transmission" in supported.evidence_text

    def test_dataset_field(self, mock_data_dir):
        claims = load_covidfact()
        assert all(c.dataset == "covidfact" for c in claims)

    def test_split_is_all(self, mock_data_dir):
        claims = load_covidfact()
        assert all(c.split == "all" for c in claims)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError, match="COVID-Fact file not found"):
            load_covidfact()


# ---------------------------------------------------------------------------
# Unified load_dataset()
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_scifact(self, mock_data_dir):
        claims = load_dataset("scifact", split="dev")
        assert len(claims) == 3
        assert all(c.dataset == "scifact" for c in claims)

    def test_pubhealth(self, mock_data_dir):
        claims = load_dataset("pubhealth", split="test")
        assert len(claims) == 4

    def test_healthver(self, mock_data_dir):
        claims = load_dataset("healthver", split="dev")
        assert len(claims) == 3

    def test_covidfact(self, mock_data_dir):
        claims = load_dataset("covidfact")
        assert len(claims) == 2

    def test_covidfact_ignores_split(self, mock_data_dir):
        claims = load_dataset("covidfact", split="train")
        assert len(claims) == 2  # split is ignored

    def test_default_splits(self, mock_data_dir):
        # Should use defaults without error
        claims_sf = load_dataset("scifact")
        assert all(c.split == "dev" for c in claims_sf)
        claims_ph = load_dataset("pubhealth")
        assert all(c.split == "test" for c in claims_ph)
        claims_hv = load_dataset("healthver")
        assert all(c.split == "dev" for c in claims_hv)

    def test_case_insensitive(self, mock_data_dir):
        claims = load_dataset("SciFact", split="dev")
        assert len(claims) == 3

    def test_unknown_dataset(self, mock_data_dir):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent")


# ---------------------------------------------------------------------------
# Filter function
# ---------------------------------------------------------------------------

class TestFilterHealthClaims:
    def test_filter_with_defaults(self, mock_data_dir):
        claims = load_dataset("pubhealth", split="test")
        filtered = filter_health_claims(claims)
        # "MMR vaccine causes autism" should match "vaccine"
        assert any("vaccine" in c.claim.lower() for c in filtered)

    def test_filter_with_custom_keywords(self, mock_data_dir):
        claims = load_dataset("pubhealth", split="test")
        filtered = filter_health_claims(claims, keywords=["smoking"])
        assert len(filtered) == 1
        assert "smoking" in filtered[0].claim.lower()

    def test_filter_no_match(self, mock_data_dir):
        claims = load_dataset("pubhealth", split="test")
        filtered = filter_health_claims(claims, keywords=["zzzznotfound"])
        assert len(filtered) == 0

    def test_filter_case_insensitive(self, mock_data_dir):
        claims = load_dataset("covidfact")
        filtered = filter_health_claims(claims, keywords=["COVID"])
        assert len(filtered) == 2  # Both claims mention COVID

    def test_filter_preserves_type(self, mock_data_dir):
        claims = load_dataset("scifact", split="dev")
        filtered = filter_health_claims(claims, keywords=["vitamin"])
        assert all(isinstance(c, BenchmarkClaim) for c in filtered)
