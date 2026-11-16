"""
Part 2 - Step 9: stream clustering with CapyMOA.
"""
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from capymoa.cluster import ClusTree, Clustream_with_kmeans, Denstream_with_dbscan
from capymoa.stream import NumpyStream

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from Online_Machine_learning_Lab.src.config import FIGURES, N_MAX, SEED, TABLES
from Online_Machine_learning_Lab.src.data import load_classification

EVAL_EVERY = 1_000
N_CLUSTERING = N_MAX
SNAPSHOTS = (5_000, 13_000, 20_000)  


# ---------------------------------------------------------------- datasets
def blobs_drift_stream(n=N_CLUSTERING):
    """Evolving 2-D Gaussian clusters with known labels (used only for evaluation).

    - clusters 0-3 move linearly across the unit square during the whole stream;
    - cluster 4 appears at 50% of the stream (cluster birth);
    - cluster 0 disappears at 75% of the stream (cluster death).
    MOA's RandomRBFGeneratorDrift was tried first but its clusters overlap so much that
    even batch k-means reaches ARI ~0.1, which makes it uninformative for evaluation.
    """
    rng = np.random.default_rng(SEED)
    start = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8], [0.5, 0.5]])
    end = np.array([[0.4, 0.3], [0.3, 0.6], [0.6, 0.3], [0.7, 0.6], [0.5, 0.9]])
    X, y = np.empty((n, 2)), np.empty(n, dtype=int)
    for t in range(n):
        active = [0, 1, 2, 3] + ([4] if t >= n // 2 else [])
        if t >= 3 * n // 4:
            active.remove(0)
        c = rng.choice(active)
        center = start[c] + (end[c] - start[c]) * t / n
        X[t] = center + rng.normal(0, 0.04, size=2)
        y[t] = c
    return NumpyStream(X, y, dataset_name="Blobs_drift", feature_names=["x1", "x2"]), 4


def online_unit_scale(X):
    """Online standardisation (rows 0..t-1 only, no look-ahead) scaled by 4*sqrt(d)
    so Euclidean distances in d dimensions have the same order of magnitude as on [0, 1]^2."""
    n = np.arange(len(X))[:, None]
    prev_sum = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)[:-1]])
    prev_sq = np.vstack([np.zeros(X.shape[1]), np.cumsum(X**2, axis=0)[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, prev_sum / n, 0.0)
        var = np.where(n > 0, prev_sq / n - mean**2, 0.0)
    std = np.sqrt(np.clip(var, 0, None))
    std[std == 0] = 1.0
    z = np.clip((X - mean) / std, -5.0, 5.0)  # prevent warm-up spikes on the first few samples
    return z / (4.0 * np.sqrt(X.shape[1]))


def insects_stream():
    data = load_classification("Insects", N_CLUSTERING)
    names = list(data[0][0])
    X = online_unit_scale(np.array([[x[k] for k in names] for x, _ in data], dtype=float))
    classes = sorted({y for _, y in data})
    y = np.array([classes.index(label) for _, label in data])
    return NumpyStream(X, y, dataset_name="Insects", feature_names=[str(k) for k in names]), len(classes)


DATASETS = {"Blobs_drift": blobs_drift_stream, "Insects": insects_stream}


def clusterers(schema, k):
    return {
        "CluStream+kMeans": Clustream_with_kmeans(schema, time_window=1000, max_num_kernels=100, k_option=k),
        "DenStream+DBSCAN": Denstream_with_dbscan(schema, horizon=1000, epsilon=0.1, mu=1, init_points=1000),
        "ClusTree+kMeans": ClusTree(schema, horizon=1000, max_height=8),
    }


# ---------------------------------------------------------------- evaluation
def current_centers(clusterer, k):
    """Macro-cluster centers. Methods exposing only micro-clusters (ClusTree in CapyMOA)
    get the usual offline step: weighted k-means on the micro-cluster centers."""
    if clusterer.implements_macro_clusters():
        centers = clusterer.get_clustering_result().get_centers()
        if len(centers) > 0:
            return np.asarray(centers, dtype=float)
        return np.empty((0, 0))
    micro = clusterer.get_micro_clustering_result()
    centers = np.asarray(micro.get_centers(), dtype=float)
    if len(centers) <= k:
        return centers
    weights = np.asarray(micro.get_weights(), dtype=float)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(centers, sample_weight=weights)
    return kmeans.cluster_centers_


def purity(y_true, y_pred):
    total = 0
    for c in np.unique(y_pred):
        total += np.bincount(y_true[y_pred == c]).max()
    return total / len(y_true)


def evaluate_window(X, y, centers):
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = dists.argmin(axis=1)
    n_used = len(np.unique(assign))
    sil = silhouette_score(X, assign) if 1 < n_used < len(X) else np.nan
    return {
        "n_clusters": len(centers),
        "silhouette": sil,
        "SSQ": dists.min(axis=1).mean(),
        "ARI": adjusted_rand_score(y, assign),
        "purity": purity(y, assign),
    }


def run_clusterer(stream, clusterer, n, k):
    stream.restart()
    window_X, window_y, records, snapshots = [], [], [], {}
    start = time.perf_counter()
    for i in range(1, n + 1):
        if not stream.has_more_instances():
            break
        instance = stream.next_instance()
        clusterer.train(instance)
        window_X.append(np.asarray(instance.x, dtype=float))
        window_y.append(instance.y_index)
        if i % EVAL_EVERY == 0:
            centers = current_centers(clusterer, k)
            X, y = np.array(window_X), np.array(window_y)
            if len(centers) > 0:
                records.append({"instance": i, **evaluate_window(X, y, centers)})
            if i in SNAPSHOTS:
                snapshots[i] = (X, y, centers)
            window_X, window_y = [], []
    elapsed = time.perf_counter() - start
    return records, snapshots, elapsed, i


# ---------------------------------------------------------------- plots
def plot_over_time(over_time):
    metrics = ["silhouette", "ARI", "purity", "n_clusters"]
    datasets = over_time["dataset"].unique()
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(16, 3.5 * len(datasets)), squeeze=False)
    for r, ds in enumerate(datasets):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            for method, grp in over_time[over_time["dataset"] == ds].groupby("method"):
                ax.plot(grp["instance"], grp[metric], marker=".", label=method)
            ax.set_title(f"{ds}: {metric}")
            ax.set_xlabel("instance")
            if metric == "n_clusters":
                ax.set_yscale("log")
        axes[r, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "clustering_over_time.png", dpi=130)
    plt.close(fig)


def plot_snapshots(all_snapshots):
    methods = list(all_snapshots)
    fig, axes = plt.subplots(len(methods), len(SNAPSHOTS), figsize=(4 * len(SNAPSHOTS), 4 * len(methods)), squeeze=False)
    for r, method in enumerate(methods):
        for c, t in enumerate(SNAPSHOTS):
            ax = axes[r, c]
            if t in all_snapshots[method]:
                X, y, centers = all_snapshots[method][t]
                ax.scatter(X[:, 0], X[:, 1], c=y, s=4, cmap="tab10", alpha=0.5)
                ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=60)
            ax.set_title(f"{method} - instance {t}")
    fig.tight_layout()
    fig.savefig(FIGURES / "clustering_snapshots_Blobs_drift.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    summary_rows, time_rows = [], []
    for ds, make_stream in DATASETS.items():
        stream, k = make_stream()
        snapshots_by_method = {}
        for method, clusterer in clusterers(stream.get_schema(), k).items():
            records, snapshots, elapsed, n_seen = run_clusterer(stream, clusterer, N_CLUSTERING, k)
            snapshots_by_method[method] = snapshots
            for rec in records:
                time_rows.append({"dataset": ds, "method": method, **rec})
            df = pd.DataFrame(records)
            summary_rows.append({
                "dataset": ds, "method": method, "n_samples": n_seen,
                **df.drop(columns="instance").mean().add_prefix("mean_").to_dict(),
                "time_s": elapsed, "latency_ms": 1000 * elapsed / n_seen,
            })
            print({k_: round(v, 4) if isinstance(v, float) else v for k_, v in summary_rows[-1].items()})
        if ds == "Blobs_drift":
            plot_snapshots(snapshots_by_method)

    summary, over_time = pd.DataFrame(summary_rows), pd.DataFrame(time_rows)
    summary.to_csv(TABLES / "capymoa_clustering_summary.csv", index=False)
    over_time.to_csv(TABLES / "capymoa_clustering_over_time.csv", index=False)
    plot_over_time(over_time)
    print(summary.round(4).to_string(index=False))
