"""
Part 1 - Steps 4 and 5: drift detection for regression and classification.
"""
import sys

import pandas as pd

from Online_Machine_learning_Lab.src.config import CLASSIFICATION_DATASETS, DRIFT_POSITIONS, FIGURES, REGRESSION_DATASETS, SYNTHETIC_DATASETS, TABLES
from Online_Machine_learning_Lab.src.data import load_classification, load_regression
from Online_Machine_learning_Lab.src.evaluation import classification_metrics, detection_delays, regression_metrics, run_with_detector
from Online_Machine_learning_Lab.src.models import classification_detectors, drift_adaptive_model, drift_base_model, regression_detectors
from Online_Machine_learning_Lab.src.plotting import plot_drifts

TASKS = {
    "regression": (REGRESSION_DATASETS, load_regression, regression_detectors, regression_metrics, "absolute error"),
    "classification": (
        CLASSIFICATION_DATASETS,
        load_classification,
        classification_detectors,
        classification_metrics,
        "error rate",
    ),
}

def drift_experiment(task):
    dataset_names, load, make_detectors, make_metrics, loss_name = TASKS[task]
    detection_rows, perf_rows = [], []

    for ds in dataset_names:
        data = load(ds)
        n = len(data)
        true_drifts = DRIFT_POSITIONS if ds in SYNTHETIC_DATASETS else ()
        make_base = lambda: drift_base_model(task)
        curves, detections = {}, {}

        # References: no drift handling, and built-in drift handling (adaptive tree).
        for name, make_model in [("No handling", make_base), ("HAT (built-in)", lambda: drift_adaptive_model(task))]:
            summary, losses, _ = run_with_detector(make_model, None, data, make_metrics(), task, False)
            perf_rows.append({"dataset": ds, "strategy": name, "n_resets": 0, **summary})
            curves[name] = losses

        for det_name, make_detector in make_detectors().items():
            # 1) Pure monitoring: where does the detector fire?
            _, _, drifts = run_with_detector(make_base, make_detector, data, make_metrics(), task, False)
            detections[det_name] = drifts
            row = {"dataset": ds, "detector": det_name, "n_detections": len(drifts),
                   "positions": " ".join(map(str, drifts[:30])) + (" ..." if len(drifts) > 30 else "")}
            if true_drifts:
                delays, false_alarms = detection_delays(drifts, true_drifts, n)
                for p, d in zip(true_drifts, delays):
                    row[f"delay@{p}"] = d
                row["false_alarms"] = false_alarms
            detection_rows.append(row)

            # 2) Drift handling: reset the model at each detection.
            summary, losses, resets = run_with_detector(make_base, make_detector, data, make_metrics(), task, True)
            perf_rows.append({"dataset": ds, "strategy": f"Reset on {det_name}", "n_resets": len(resets), **summary})
            curves[f"Reset on {det_name}"] = losses

        print(f"[{task}] {ds}: " + ", ".join(f"{k}={len(v)}" for k, v in detections.items()))
        plot_drifts(ds, curves, detections, true_drifts, loss_name, FIGURES / f"drift_{task}_{ds}.png")

    det_table, perf_table = pd.DataFrame(detection_rows), pd.DataFrame(perf_rows)
    det_table.to_csv(TABLES / f"drift_{task}_detections.csv", index=False)
    perf_table.to_csv(TABLES / f"drift_{task}_performance.csv", index=False)
    return det_table, perf_table


if __name__ == "__main__":
    for task in sys.argv[1:] or list(TASKS):
        print(f"\n===== Drift detection: {task} =====")
        det_table, perf_table = drift_experiment(task)
        print(det_table.drop(columns="positions").to_string(index=False))
        print(perf_table.round(4).to_string(index=False))
