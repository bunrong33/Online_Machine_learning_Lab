"""Prequential (test-then-train) evaluation loops for River models."""
import math
import time

from river import metrics, stats


def regression_metrics():
    return {"MAE": metrics.MAE(), "RMSE": metrics.RMSE(), "R2": metrics.R2()}


def classification_metrics():
    # Macro-F1 is used for every dataset so binary and multi-class streams are treated the same.
    return {"Accuracy": metrics.Accuracy(), "MacroF1": metrics.MacroF1(), "Kappa": metrics.CohenKappa()}


def instance_loss(task, y, y_pred):
    """Per-instance loss used for curves: absolute error or 0/1 error."""
    if task == "regression":
        return abs(y - y_pred)
    return float(y_pred != y)


def update_metrics(metric_dict, y, y_pred):
    # A classifier returns None before it has seen any label: skip metrics for that sample.
    if y_pred is not None:
        for m in metric_dict.values():
            m.update(y, y_pred)


def run_prequential(model, data, metric_dict, task):
    """Test-then-train over `data`. Returns (summary dict, list of per-instance losses)."""
    losses = []
    start = time.perf_counter()
    for x, y in data:
        y_pred = model.predict_one(x)  # 1) test
        update_metrics(metric_dict, y, y_pred)
        losses.append(instance_loss(task, y, y_pred))
        model.learn_one(x, y)  # 2) train
    elapsed = time.perf_counter() - start

    summary = {name: m.get() for name, m in metric_dict.items()}
    summary["time_s"] = elapsed
    summary["latency_ms"] = 1000 * elapsed / len(data)
    summary["memory_kb"] = model._raw_memory_usage / 1024  # final model size
    return summary, losses


def run_with_detector(make_model, make_detector, data, metric_dict, task, reset_on_drift):
    """Prequential run where a drift detector monitors the model's error.

    - make_detector=None          -> plain model, no drift handling (baseline)
    - reset_on_drift=False        -> detector only *monitors* (positions of drifts)
    - reset_on_drift=True         -> the model is replaced by a fresh one at each drift

    For regression the absolute error is divided by the running std of the target,
    so detector thresholds are comparable across datasets with different scales.
    """
    model = make_model()
    detector = make_detector() if make_detector else None
    target_var = stats.Var()
    drifts, losses = [], []

    start = time.perf_counter()
    for i, (x, y) in enumerate(data):
        y_pred = model.predict_one(x)
        update_metrics(metric_dict, y, y_pred)
        loss = instance_loss(task, y, y_pred)
        losses.append(loss)

        if detector is not None:
            if task == "regression":
                scale = math.sqrt(target_var.get()) or 1.0
                signal = loss / scale
            else:
                signal = loss
            detector.update(signal)
            if detector.drift_detected:
                drifts.append(i)
                if reset_on_drift:
                    model = make_model()

        if task == "regression":
            target_var.update(y)
        model.learn_one(x, y)
    elapsed = time.perf_counter() - start

    summary = {name: m.get() for name, m in metric_dict.items()}
    summary["time_s"] = elapsed
    summary["latency_ms"] = 1000 * elapsed / len(data)
    return summary, losses, drifts


def detection_delays(detections, true_drifts, n):
    """For each true drift, delay of the first detection before the next true drift.

    Returns (list of delays or None when missed, number of false alarms).
    False alarms = detections that are not the first detection after a true drift.
    """
    bounds = list(true_drifts) + [n]
    delays, matched = [], set()
    for start, end in zip(bounds[:-1], bounds[1:]):
        hits = [d for d in detections if start <= d < end]
        delays.append(hits[0] - start if hits else None)
        if hits:
            matched.add(hits[0])
    return delays, len(detections) - len(matched)
