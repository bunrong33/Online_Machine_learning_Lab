"""Dataset loading.

Each stream is materialised once as a list of (x, y) pairs. This guarantees that all
models (and both River and CapyMOA) are evaluated on identical samples in identical order.
Feature engineering is stateless (done per sample), so there is no data leakage.
"""
import itertools
import math

from river import datasets
from river.datasets import synth

from Online_Machine_learning_Lab.src.config import DRIFT_POSITIONS, N_MAX, SEED

# Width of the drift transition. River computes exp(-4 * (t - position) / width), which
# overflows for very small widths, so 50 instances is used (a near-abrupt drift).
SEA_DRIFT_WIDTH = 50


# ---------------------------------------------------------------- feature helpers
def bikes_features(x):
    """Numeric weather features + time of day + one-hot station."""
    moment = x["moment"]
    features = {k: x[k] for k in ("clouds", "humidity", "pressure", "temperature", "wind")}
    features["hour"] = moment.hour
    features["weekday"] = moment.weekday()
    features[f"station={x['station']}"] = 1.0
    return features


def taxis_features(x):
    """Coordinates, straight-line distance, time of day and passenger count."""
    t = x["pickup_datetime"]
    d_lat = x["dropoff_latitude"] - x["pickup_latitude"]
    d_lon = x["dropoff_longitude"] - x["pickup_longitude"]
    return {
        "vendor_id": float(x["vendor_id"]),
        "passenger_count": float(x["passenger_count"]),
        "pickup_latitude": x["pickup_latitude"],
        "pickup_longitude": x["pickup_longitude"],
        "dropoff_latitude": x["dropoff_latitude"],
        "dropoff_longitude": x["dropoff_longitude"],
        "distance": math.sqrt(d_lat**2 + d_lon**2),
        "hour": t.hour,
        "weekday": t.weekday(),
    }


def to_list(stream, n_max, transform=None, target=lambda y: y):
    """Take the first n_max samples of a River stream as a list of (x, y)."""
    data = []
    for x, y in itertools.islice(stream, n_max):
        x = transform(x) if transform else dict(x)
        data.append((x, target(y)))
    return data


# ---------------------------------------------------------------- regression
def load_regression(name, n_max=N_MAX):
    if name == "Bikes":
        return to_list(datasets.Bikes(), n_max, bikes_features, float)
    if name == "Taxis":
        return to_list(datasets.Taxis(), n_max, taxis_features, float)
    if name == "TrumpApproval":
        return to_list(datasets.TrumpApproval(), n_max, target=float)
    if name == "FriedmanDrift":
        # 'gra' = global recurring abrupt drift: concept changes at position[0]
        # and comes back to the original concept at position[1].
        stream = synth.FriedmanDrift(drift_type="gra", position=DRIFT_POSITIONS, seed=SEED)
        return to_list(stream, n_max, target=float)
    raise ValueError(f"Unknown regression dataset: {name}")


# ---------------------------------------------------------------- classification
def sea_concept_drift_stream():
    """SEA stream with two abrupt drifts (variant 0 -> 1 -> 2) at DRIFT_POSITIONS.

    ConceptDriftStream only advances the inner stream once it is being used, so the
    inner drift position is relative to the first drift.
    """
    p1, p2 = DRIFT_POSITIONS
    second_part = synth.ConceptDriftStream(
        stream=synth.SEA(variant=1, seed=SEED),
        drift_stream=synth.SEA(variant=2, seed=SEED),
        position=p2 - p1,
        width=SEA_DRIFT_WIDTH,
        seed=SEED,
    )
    return synth.ConceptDriftStream(
        stream=synth.SEA(variant=0, seed=SEED),
        drift_stream=second_part,
        position=p1,
        width=SEA_DRIFT_WIDTH,
        seed=SEED,
    )


def load_classification(name, n_max=N_MAX):
    if name == "Elec2":
        return to_list(datasets.Elec2(), n_max)
    if name == "Insects":
        # River 0.26 yields the 33 numeric sensor columns of Insects as strings.
        return to_list(
            datasets.Insects(variant="abrupt_balanced"),
            n_max,
            transform=lambda x: {k: float(v) for k, v in x.items()},
        )
    if name == "Phishing":
        return to_list(datasets.Phishing(), n_max)
    if name == "SEA_drift":
        return to_list(sea_concept_drift_stream(), n_max)
    raise ValueError(f"Unknown classification dataset: {name}")
