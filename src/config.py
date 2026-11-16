"""Shared configuration for all experiments (seeds, stream sizes, paths)."""
from pathlib import Path

SEED = 42

# Large streams are truncated to the first N_MAX samples to keep runtimes reasonable.
# Every model and both libraries see exactly the same samples in the same order.
N_MAX = 20_000

# Known (ground-truth) drift positions used for the synthetic streams.
DRIFT_POSITIONS = (7_000, 14_000)

REGRESSION_DATASETS = ["Bikes", "Taxis", "TrumpApproval", "FriedmanDrift"]
CLASSIFICATION_DATASETS = ["Elec2", "Insects", "Phishing", "SEA_drift"]
SYNTHETIC_DATASETS = {"FriedmanDrift", "SEA_drift"}

# Window used to plot rolling metrics over time.
ROLLING_WINDOW = 500

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
