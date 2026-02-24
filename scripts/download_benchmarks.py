"""Download benchmark datasets for evaluation.

Downloads 4 datasets used to evaluate the health claim checker:
  - SciFact: 1,409 claims (SUPPORT/CONTRADICT)
  - PUBHEALTH: 11,832 claims (true/false/mixture/unproven)
  - HealthVer: 14,330 claim-evidence pairs (SUPPORT/REFUTE/NEUTRAL)
  - COVID-Fact: 4,086 claims (SUPPORTED/REFUTED)

Usage:
    uv run python scripts/download_benchmarks.py              # download all
    uv run python scripts/download_benchmarks.py --dataset scifact
    uv run python scripts/download_benchmarks.py --dataset pubhealth
    uv run python scripts/download_benchmarks.py --dataset healthver
    uv run python scripts/download_benchmarks.py --dataset covidfact
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"

# ---------------------------------------------------------------------------
# SciFact
# ---------------------------------------------------------------------------

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_EXPECTED = ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl", "corpus.jsonl"]


def download_scifact() -> None:
    """Download SciFact dataset from S3 tar.gz archive."""
    dest = DATA_DIR / "scifact"
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((dest / f).exists() for f in SCIFACT_EXPECTED):
        print("SciFact: already downloaded, skipping.")
        return

    print("Downloading SciFact...")
    resp = requests.get(SCIFACT_URL, stream=True, timeout=120)
    resp.raise_for_status()

    # Write to temp file then extract
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)

    try:
        with tarfile.open(tmp_path, "r:gz") as tar:
            # Files are inside a top-level "data/" directory in the archive
            for member in tar.getmembers():
                if member.isfile():
                    # Strip the leading directory (e.g. "data/claims_train.jsonl" -> "claims_train.jsonl")
                    member.name = Path(member.name).name
                    tar.extract(member, dest)
        print(f"SciFact: extracted to {dest}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Verify
    missing = [f for f in SCIFACT_EXPECTED if not (dest / f).exists()]
    if missing:
        print(f"  WARNING: missing expected files: {missing}")
    else:
        print(f"  OK: {len(SCIFACT_EXPECTED)} files verified.")


# ---------------------------------------------------------------------------
# PUBHEALTH
# ---------------------------------------------------------------------------

PUBHEALTH_GDRIVE_ID = "1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj"
PUBHEALTH_EXPECTED = ["train.tsv", "dev.tsv", "test.tsv"]


def _download_google_drive(file_id: str, dest_path: Path) -> None:
    """Download a file from Google Drive, handling the large-file confirmation page."""
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    resp = session.get(url, params={"id": file_id}, stream=True, timeout=120)
    resp.raise_for_status()

    # Google may show a virus-scan warning page with a confirm token
    confirm_token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        resp = session.get(
            url,
            params={"id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def download_pubhealth() -> None:
    """Download PUBHEALTH dataset from Google Drive zip."""
    dest = DATA_DIR / "pubhealth"
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((dest / f).exists() for f in PUBHEALTH_EXPECTED):
        print("PUBHEALTH: already downloaded, skipping.")
        return

    print("Downloading PUBHEALTH from Google Drive...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _download_google_drive(PUBHEALTH_GDRIVE_ID, tmp_path)

        # Verify it's actually a zip (Google may return an HTML error page)
        if not zipfile.is_zipfile(tmp_path):
            print("PUBHEALTH: ERROR — downloaded file is not a valid zip.")
            print("  This may mean the Google Drive link has changed or requires manual auth.")
            print(f"  Try downloading manually: https://drive.google.com/uc?id={PUBHEALTH_GDRIVE_ID}")
            return

        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Extract TSV files, stripping any directory prefix
            for info in zf.infolist():
                if info.is_dir():
                    continue
                filename = Path(info.filename).name
                if filename.endswith(".tsv"):
                    info.filename = filename
                    zf.extract(info, dest)

        print(f"PUBHEALTH: extracted to {dest}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Verify
    missing = [f for f in PUBHEALTH_EXPECTED if not (dest / f).exists()]
    if missing:
        print(f"  WARNING: missing expected files: {missing}")
    else:
        print(f"  OK: {len(PUBHEALTH_EXPECTED)} files verified.")


# ---------------------------------------------------------------------------
# HealthVer
# ---------------------------------------------------------------------------

HEALTHVER_REPO = "https://github.com/sarrouti/HealthVer.git"
HEALTHVER_EXPECTED = ["healthver_train.csv", "healthver_dev.csv", "healthver_test.csv"]


def download_healthver() -> None:
    """Download HealthVer dataset by shallow-cloning the GitHub repo."""
    dest = DATA_DIR / "healthver"
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if all((dest / f).exists() for f in HEALTHVER_EXPECTED):
        print("HealthVer: already downloaded, skipping.")
        return

    print("Downloading HealthVer via git clone...")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", HEALTHVER_REPO, tmpdir],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            print("HealthVer: ERROR — git is not installed.")
            return
        except subprocess.CalledProcessError as e:
            print(f"HealthVer: ERROR — git clone failed: {e.stderr}")
            return

        # Copy data files from the cloned repo
        repo_data = Path(tmpdir) / "data"
        if not repo_data.exists():
            # Try alternate locations
            for candidate in [Path(tmpdir), Path(tmpdir) / "dataset"]:
                csvs = list(candidate.glob("*.csv"))
                if csvs:
                    repo_data = candidate
                    break

        if repo_data.exists():
            for f in repo_data.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
            print(f"HealthVer: extracted to {dest}")
        else:
            print("HealthVer: WARNING — could not find data files in cloned repo.")

    # Verify
    found = list(dest.glob("*.csv"))
    if found:
        print(f"  OK: {len(found)} CSV files found.")
    else:
        print("  WARNING: no CSV files found after extraction.")


# ---------------------------------------------------------------------------
# COVID-Fact
# ---------------------------------------------------------------------------

COVIDFACT_URL = "https://raw.githubusercontent.com/asaakyan/covidfact/master/COVIDFACT_dataset.jsonl"
COVIDFACT_FILE = "COVIDFACT_dataset.jsonl"


def download_covidfact() -> None:
    """Download COVID-Fact dataset (single JSONL file from GitHub)."""
    dest = DATA_DIR / "covidfact"
    dest.mkdir(parents=True, exist_ok=True)
    dest_file = dest / COVIDFACT_FILE

    # Check if already downloaded
    if dest_file.exists() and dest_file.stat().st_size > 0:
        print("COVID-Fact: already downloaded, skipping.")
        return

    print("Downloading COVID-Fact...")
    resp = requests.get(COVIDFACT_URL, timeout=60)
    resp.raise_for_status()

    dest_file.write_text(resp.text, encoding="utf-8")
    print(f"COVID-Fact: saved to {dest_file}")

    lines = resp.text.strip().splitlines()
    print(f"  OK: {len(lines)} entries.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DOWNLOADERS = {
    "scifact": download_scifact,
    "pubhealth": download_pubhealth,
    "healthver": download_healthver,
    "covidfact": download_covidfact,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets for evaluation.")
    parser.add_argument(
        "--dataset",
        choices=[*DOWNLOADERS.keys(), "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "all":
        for name, fn in DOWNLOADERS.items():
            try:
                fn()
            except Exception as e:
                print(f"ERROR downloading {name}: {e}")
            print()
    else:
        try:
            DOWNLOADERS[args.dataset]()
        except Exception as e:
            print(f"ERROR downloading {args.dataset}: {e}")
            sys.exit(1)

    print("Benchmark download complete!")


if __name__ == "__main__":
    main()
