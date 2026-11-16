"""River model and drift-detector factories.

Factories (functions returning a fresh model) are used so that every run starts
from an untrained model, and so that a model can be re-created after a drift.
"""
from river import drift, forest, linear_model, multiclass, naive_bayes, neighbors, optim, preprocessing, tree

from Online_Machine_learning_Lab.src.config import SEED

KNN_WINDOW = 500  # number of recent samples kept by the KNN search engine


# ---------------------------------------------------------------- regression
def scaled_regressor(regressor):
    """Online feature scaling + online target scaling (both learned incrementally)."""
    return preprocessing.TargetStandardScaler(regressor=preprocessing.StandardScaler() | regressor)


def regression_models():
    return {
        "HoeffdingTree": lambda: scaled_regressor(tree.HoeffdingTreeRegressor(grace_period=200)),
        "ARF": lambda: scaled_regressor(forest.ARFRegressor(n_models=10, seed=SEED)),
        "LinearReg(Adam)": lambda: scaled_regressor(
            linear_model.LinearRegression(optimizer=optim.Adam(lr=0.01))
        ),
        "KNN": lambda: scaled_regressor(
            neighbors.KNNRegressor(n_neighbors=5, engine=neighbors.LazySearch(window_size=KNN_WINDOW))
        ),
    }


# ---------------------------------------------------------------- classification
def classification_models():
    return {
        "HoeffdingTree": lambda: tree.HoeffdingTreeClassifier(grace_period=200),
        "ARF": lambda: forest.ARFClassifier(n_models=10, seed=SEED),
        # One-vs-rest makes logistic regression usable on multi-class streams (Insects).
        "LogReg(Adam)": lambda: preprocessing.StandardScaler()
        | multiclass.OneVsRestClassifier(linear_model.LogisticRegression(optimizer=optim.Adam(lr=0.01))),
        "GaussianNB": lambda: naive_bayes.GaussianNB(),
    }


# ---------------------------------------------------------------- drift detection
def regression_detectors():
    """Detectors monitoring the (normalised) absolute error of a regressor."""
    return {
        "ADWIN": lambda: drift.ADWIN(),
        "PageHinkley": lambda: drift.PageHinkley(mode="up"),  # only error increases
        "KSWIN": lambda: drift.KSWIN(seed=SEED),
    }


def classification_detectors():
    """Detectors monitoring the 0/1 error of a classifier."""
    return {
        "ADWIN": lambda: drift.ADWIN(),
        "DDM": lambda: drift.binary.DDM(),
        "EDDM": lambda: drift.binary.EDDM(),
        "HDDM_A": lambda: drift.binary.HDDMA(),  # named HDDMA in recent River versions
    }


# Base learners used in the drift experiments (no internal drift handling) and
# their adaptive counterparts with built-in drift handling (used as a reference).
def drift_base_model(task):
    if task == "regression":
        return scaled_regressor(tree.HoeffdingTreeRegressor(grace_period=200))
    return tree.HoeffdingTreeClassifier(grace_period=200)


def drift_adaptive_model(task):
    if task == "regression":
        return scaled_regressor(tree.HoeffdingAdaptiveTreeRegressor(grace_period=200, seed=SEED))
    return tree.HoeffdingAdaptiveTreeClassifier(grace_period=200, seed=SEED)
