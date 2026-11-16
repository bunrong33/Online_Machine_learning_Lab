"""
Part 2 -Step 8: CapyMOA classification, compared with River.

"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from capymoa.classifier import AdaptiveRandomForestClassifier, HoeffdingTree, NaiveBayes, SGDClassifier
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import NumpyStream

from Online_Machine_learning_Lab.src.config import CLASSIFICATION_DATASETS, FIGURES, SEED, TABLES
from Online_Machine_learning_Lab.src.data import load_classification
from Online_Machine_learning_Lab.src.evaluation import classification_metrics


# ---------------------------------------------------------------- data conversion
def to_numpy(data):
    """List of (dict, label) -> X (float matrix), y (int codes), feature names, classes."""
    feature_names = list(dict.fromkeys(k for x, _ in data for k in x))
    X = np.array([[x.get(k, 0.0) for k in feature_names] for x, _ in data], dtype=float)
    classes = sorted({y for _, y in data}, key=str)
    y = np.array([classes.index(label) for _, label in data])
    return X, y, [str(k) for k in feature_names], classes


def online_standardize(X):
    """Standardise row t with the mean/std of rows 0..t-1 only (no look-ahead).

    This reproduces River's online StandardScaler for the linear model, which CapyMOA's
    SGDClassifier (a scikit-learn wrapper) does not include.
    """
    n = np.arange(len(X))[:, None]
    prev_sum = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)[:-1]])
    prev_sq = np.vstack([np.zeros(X.shape[1]), np.cumsum(X**2, axis=0)[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, prev_sum / n, 0.0)
        var = np.where(n > 0, prev_sq / n - mean**2, 0.0)
    std = np.sqrt(np.clip(var, 0, None))
    std[std == 0] = 1.0
    return (X - mean) / std


def capymoa_models(schema, n_features):
    sqrt_fraction = math.floor(math.sqrt(n_features)) / n_features  # River's max_features="sqrt"
    return {
        "HoeffdingTree": (HoeffdingTree(schema, random_seed=SEED, grace_period=200, confidence=1e-7), False),
        "ARF": (
            AdaptiveRandomForestClassifier(schema, random_seed=SEED, ensemble_size=10, max_features=sqrt_fraction),
            False,
        ),
        "LogReg(SGD)": (
            SGDClassifier(schema, loss="log_loss", learning_rate="constant", eta0=0.01, random_seed=SEED),
            True,  # needs scaled inputs
        ),
        "GaussianNB": (NaiveBayes(schema, random_seed=SEED), False),
    }


def score_predictions(y_true, y_pred):
    metric_dict = classification_metrics()
    for yt, yp in zip(y_true, y_pred):
        if yp is not None:
            for m in metric_dict.values():
                m.update(int(yt), int(yp))
    return {name: m.get() for name, m in metric_dict.items()}


def run_capymoa():
    rows = []
    for ds in CLASSIFICATION_DATASETS:
        data = load_classification(ds)
        X, y, names, _ = to_numpy(data)
        streams = {
            False: NumpyStream(X, y, dataset_name=ds, feature_names=names, target_type="categorical"),
            True: NumpyStream(online_standardize(X), y, dataset_name=ds, feature_names=names, target_type="categorical"),
        }
        schema = streams[False].get_schema()
        for model_name, (learner, scaled) in capymoa_models(schema, X.shape[1]).items():
            results = prequential_evaluation(
                streams[scaled], learner, window_size=500, store_predictions=True, store_y=True
            )
            scores = score_predictions(results.ground_truth_y(), results.predictions())
            wallclock = results.wallclock()
            rows.append({
                "dataset": ds, "model": model_name, "n_samples": len(data), **scores,
                "time_s": wallclock, "cpu_time_s": results.cpu_time(), "latency_ms": 1000 * wallclock / len(data),
            })
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "capymoa_classification.csv", index=False)
    return table


def compare_with_river(capy):
    river = pd.read_csv(TABLES / "river_classification.csv")
    river["model"] = river["model"].replace({"LogReg(Adam)": "LogReg"})
    capy = capy.assign(model=capy["model"].replace({"LogReg(SGD)": "LogReg"}))
    cols = ["dataset", "model", "Accuracy", "MacroF1", "Kappa", "time_s", "latency_ms"]
    merged = river[cols].merge(capy[cols], on=["dataset", "model"], suffixes=("_river", "_capymoa"))
    merged["speedup_capymoa"] = merged["time_s_river"] / merged["time_s_capymoa"]
    merged.to_csv(TABLES / "river_vs_capymoa.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = merged["dataset"] + "\n" + merged["model"]
    x = np.arange(len(merged))
    for ax, metric, ylabel in [(axes[0], "Accuracy", "accuracy"), (axes[1], "latency_ms", "latency (ms / instance, log)")]:
        ax.bar(x - 0.2, merged[f"{metric}_river"], 0.4, label="River")
        ax.bar(x + 0.2, merged[f"{metric}_capymoa"], 0.4, label="CapyMOA")
        ax.set_xticks(x, labels, rotation=90, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale("log")
    fig.suptitle("River vs CapyMOA (same streams, prequential evaluation)")
    fig.tight_layout()
    fig.savefig(FIGURES / "river_vs_capymoa.png", dpi=130)
    plt.close(fig)
    return merged


if __name__ == "__main__":
    capy = run_capymoa()
    print(capy.round(4).to_string(index=False))
    print(compare_with_river(capy).round(4).to_string(index=False))
