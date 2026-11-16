"""
Part 1 - Steps 2, 3 and 5: compare 4 River methods on 4 datasets 

"""
import sys

import pandas as pd

from Online_Machine_learning_Lab.src.config import CLASSIFICATION_DATASETS, FIGURES, REGRESSION_DATASETS, TABLES
from Online_Machine_learning_Lab.src.data import load_classification, load_regression
from Online_Machine_learning_Lab.src.evaluation import classification_metrics, regression_metrics, run_prequential
from Online_Machine_learning_Lab.src.models import classification_models, regression_models
from Online_Machine_learning_Lab.src.plotting import plot_rolling_losses

TASKS = {
    "regression": (REGRESSION_DATASETS, load_regression, regression_models, regression_metrics, "absolute error"),
    "classification": (
        CLASSIFICATION_DATASETS,
        load_classification,
        classification_models,
        classification_metrics,
        "error rate",
    ),
}


def compare(task):
    dataset_names, load, make_models, make_metrics, loss_name = TASKS[task]
    rows, curves = [], {}
    for ds in dataset_names:
        data = load(ds)
        for model_name, make_model in make_models().items():
            summary, losses = run_prequential(make_model(), data, make_metrics(), task)
            rows.append({"dataset": ds, "model": model_name, "n_samples": len(data), **summary})
            curves[(ds, model_name)] = losses
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / f"river_{task}.csv", index=False)
    plot_rolling_losses(
        curves, dataset_names, loss_name, f"River {task}: prequential {loss_name}",
        FIGURES / f"river_{task}_rolling.png",
    )
    return table


if __name__ == "__main__":
    for task in sys.argv[1:] or list(TASKS):
        print(f"\n===== River {task} =====")
        print(compare(task).round(4).to_string(index=False))
