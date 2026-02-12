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
