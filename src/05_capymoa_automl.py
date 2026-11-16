"""
Part 2 - Step 10 (bonus): CapyMOA AutoML.

Outputs: results/tables/capymoa_automl.csv
"""
import importlib

import pandas as pd
from capymoa.automl import BanditClassifier, EpsilonGreedy, SuccessiveHalvingClassifier
from capymoa.classifier import AdaptiveRandomForestClassifier, HoeffdingAdaptiveTree, HoeffdingTree, NaiveBayes
from capymoa.evaluation import prequential_evaluation
from capymoa.stream import NumpyStream

from Online_Machine_learning_Lab.src.config import SEED, TABLES
from Online_Machine_learning_Lab.src.data import load_classification

# Reuse the helpers of the CapyMOA classification script (file names start with a digit).
capy_clf = importlib.import_module("03_capymoa_classification")

AUTOML_DATASETS = ["Elec2", "SEA_drift"]


def candidates(schema):
    return {
        "HT(gp=50)": HoeffdingTree(schema, random_seed=SEED, grace_period=50, confidence=1e-7),
        "HT(gp=200)": HoeffdingTree(schema, random_seed=SEED, grace_period=200, confidence=1e-7),
        "HAT": HoeffdingAdaptiveTree(schema, random_seed=SEED),
        "NaiveBayes": NaiveBayes(schema, random_seed=SEED),
        "ARF(10)": AdaptiveRandomForestClassifier(schema, random_seed=SEED, ensemble_size=10),
    }


def evaluate(stream, learner, n):
    results = prequential_evaluation(stream, learner, window_size=1000, store_predictions=True, store_y=True)
    scores = capy_clf.score_predictions(results.ground_truth_y(), results.predictions())
    return {**scores, "time_s": results.wallclock(), "latency_ms": 1000 * results.wallclock() / n}


if __name__ == "__main__":
    rows = []
    for ds in AUTOML_DATASETS:
        data = load_classification(ds)
        X, y, names, _ = capy_clf.to_numpy(data)
        stream = NumpyStream(X, y, dataset_name=ds, feature_names=names, target_type="categorical")
        schema, n = stream.get_schema(), len(data)

        # Each candidate alone (fresh instances).
        for name, model in candidates(schema).items():
            rows.append({"dataset": ds, "approach": f"single: {name}", **evaluate(stream, model, n)})
            print(rows[-1])

        # AutoML approaches over the same pool of candidates (fresh instances again).
        automl = {
            "AutoML: Bandit (eps-greedy)": BanditClassifier(
                schema, random_seed=SEED, base_classifiers=list(candidates(schema).values()),
                policy=EpsilonGreedy(epsilon=0.1, burn_in=100),
            ),
            "AutoML: Successive Halving": SuccessiveHalvingClassifier(
                schema, random_seed=SEED, base_classifiers=list(candidates(schema).values()), max_instances=n,
            ),
        }
        for name, learner in automl.items():
            rows.append({"dataset": ds, "approach": name, **evaluate(stream, learner, n)})
            print(rows[-1])

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "capymoa_automl.csv", index=False)
    print(table.round(4).to_string(index=False))
