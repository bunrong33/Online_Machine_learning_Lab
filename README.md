# Online Machine Learning with River and CapyMOA

Implementation of the lab *"Online Machine Learning with River and CapyMOA"* (M2 Data Science, IP Paris).

## Environment
- Python 3.14, River 0.26.1, CapyMOA 0.15.1 (OpenJDK 17), pandas, matplotlib, scikit-learn.
- Activate the venv: `source .venv/bin/activate`

## Run order
Run the scripts from the project root:

```bash
python src/01_river_comparison.py          # Steps 2, 3, 5: 4 regressors and 4 classifiers on 4 datasets each
python src/02_river_drift.py               # Steps 4, 5: drift detectors, delays, with vs without drift handling
python src/03_capymoa_classification.py    # Step 8: the same classifiers in CapyMOA, compared with River (needs 01)
python src/04_capymoa_clustering.py        # Step 9: stream clustering (CluStream, DenStream, ClusTree)
python src/05_capymoa_automl.py            # Step 10 (bonus): Bandit and Successive Halving AutoML
```
`01` and `02` take `regression` or `classification` as an optional argument.
Real datasets (Bikes, Taxis, Elec2, Insects) are downloaded by River the first time they are used.

## Project layout
| File | Content |
|---|---|
| `src/config.py` | seed (42), stream size (`N_MAX = 20 000`), true drift positions, paths |
| `src/data.py` | dataset loading and stateless feature engineering, synthetic streams |
| `src/models.py` | River models and drift detectors (factories) |
| `src/evaluation.py` | prequential (test-then-train) loops, drift detection loop, detection delays |
| `src/plotting.py` | rolling-metric and drift plots |
| `results/tables/*.csv` | all result tables |
| `results/figures/*.png` | all figures |
| `results/analysis.md` | detailed Markdown interpretation of all experimental results |

## Methodology (summary)
- **Prequential evaluation**: every instance is first used to test the model, then to train it.
- **Same data for everyone**: each stream is loaded once as a list, so every model and both libraries see the same samples in the same order. Large streams are cut to their first 20 000 instances.
- **No leakage**: features are built from each sample alone. Scaling is done online (`StandardScaler`, `TargetStandardScaler` in River; in CapyMOA, each row is scaled using only the rows before it).
- **Seeds**: `SEED = 42` for every stochastic model and generator.
- **Cost**: total time, latency (ms per instance, predict + learn) and final model memory (River, `_raw_memory_usage`).

### Methods
| Type | Regression (River) | Classification (River) | Classification (CapyMOA) |
|---|---|---|---|
| Tree | HoeffdingTreeRegressor | HoeffdingTreeClassifier | HoeffdingTree (confidence=1e-7 as in River) |
| Ensemble | ARFRegressor (10 trees) | ARFClassifier (10 trees) | AdaptiveRandomForest (10 trees, sqrt features) |
| Linear, adaptive LR | LinearRegression + Adam | LogisticRegression + Adam (one-vs-rest) | SGDClassifier (log loss, constant LR)* |
| Other | KNNRegressor (k=5, window 500) | GaussianNB | NaiveBayes (Gaussian for numeric features) |

\*CapyMOA has no linear model trained with Adam; the closest one is scikit-learn's SGD with logistic loss.

### Datasets
- Regression: Bikes, Taxis, TrumpApproval (real), FriedmanDrift `gra` with abrupt drifts at 7 000 and 14 000 (synthetic).
- Classification: Elec2, Insects (abrupt_balanced), Phishing (real), SEA + ConceptDriftStream (variant 0 → 1 → 2) with drifts at 7 000 and 14 000 (synthetic).
- Clustering: Blobs_drift (synthetic 2-D clusters that move, appear and disappear) and Insects (labels used only for evaluation).


---




