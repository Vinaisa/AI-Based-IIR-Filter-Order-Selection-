
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

from S3model import engineer_features

DATA = "iir_filter_dataset.csv"
RESULT_FILE = "hybrid_comparison.csv"
PREDICTION_FILE = "hybrid_test_predictions.csv"

RANDOM_STATE = 42

# Load data and generate the same features as S3model.py.
raw = pd.read_csv(DATA)
df, X, _, _ = engineer_features(raw)

y = df["reference_order"].astype(int)
groups = df["spec_id"]

# Same fixed grouped split used by S3model.py and S3tune.py.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=RANDOM_STATE
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
g_train = groups.iloc[train_idx]

assert set(g_train).isdisjoint(set(groups.iloc[test_idx]))

# Tune the regressor using training data only.
regressor_search = GridSearchCV(
    ExtraTreesRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    {
        "n_estimators": [500, 700],
        "max_features": [0.7, 1.0],
        "min_samples_leaf": [1, 2, 3],
    },
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=3),
    n_jobs=-1,
)
regressor_search.fit(X_train, y_train, groups=g_train)
regressor = regressor_search.best_estimator_

# Classifier predicts one of the observed integer order classes.
classifier = ExtraTreesClassifier(
    n_estimators=700,
    max_features=0.7,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
classifier.fit(X_train, y_train)

reg_raw = regressor.predict(X_test)
reg_pred = np.clip(np.rint(reg_raw).astype(int), 1, 100)

class_pred = classifier.predict(X_test).astype(int)

# Hybrid prediction: blend regression and classification estimates.
# The blend weight is fixed in advance, not selected on the test set.
hybrid_pred = np.clip(
    np.rint(0.5 * reg_raw + 0.5 * class_pred).astype(int),
    1, 100,
)

actual = y_test.to_numpy(dtype=int)

def calculate_metrics(name, predictions):
    errors = np.abs(actual - predictions)
    return {
        "Model": name,
        "Exact accuracy (%)": round(
            np.mean(errors == 0) * 100, 2
        ),
        "Within ±1 (%)": round(
            np.mean(errors <= 1) * 100, 2
        ),
        "MAE": round(
            mean_absolute_error(actual, predictions), 4
        ),
    }

results = [
    calculate_metrics("Tuned Extra Trees Regressor", reg_pred),
    calculate_metrics("Extra Trees Classifier", class_pred),
    calculate_metrics("Hybrid 50/50", hybrid_pred),
]

results_df = pd.DataFrame(results).sort_values(
    ["Exact accuracy (%)", "MAE"],
    ascending=[False, True],
)
results_df.to_csv(RESULT_FILE, index=False)

# Save predictions for inspection; do not overwrite production artifacts.
out = df.iloc[test_idx][
    [
        "spec_id",
        "filter_response",
        "filter_type",
        "design_method",
        "reference_order",
    ]
].copy()

out.rename(columns={"reference_order": "actual_order"}, inplace=True)
out["regressor_prediction"] = reg_pred
out["classifier_prediction"] = class_pred
out["hybrid_prediction"] = hybrid_pred
out["regressor_error"] = np.abs(
    out["actual_order"].to_numpy() - reg_pred
)
out["classifier_error"] = np.abs(
    out["actual_order"].to_numpy() - class_pred
)
out["hybrid_error"] = np.abs(
    out["actual_order"].to_numpy() - hybrid_pred
)
out.to_csv(PREDICTION_FILE, index=False)

print("\nBest regressor parameters:")
print(regressor_search.best_params_)
print("Grouped CV MAE:", round(-regressor_search.best_score_, 4))

print("\n--- TEST SET COMPARISON ---")
print(results_df.to_string(index=False))

print("\n--- RESULTS BY RESPONSE ---")
for response, subset in out.groupby("filter_response"):
    print(f"\n{response}")
    for label, error_col in [
        ("Regressor", "regressor_error"),
        ("Classifier", "classifier_error"),
        ("Hybrid", "hybrid_error"),
    ]:
        errors = subset[error_col].to_numpy()
        print(
            f"  {label}: exact={np.mean(errors == 0) * 100:.2f}%, "
            f"within ±1={np.mean(errors <= 1) * 100:.2f}%, "
            f"MAE={np.mean(errors):.4f}"
        )

print("\nSaved:", RESULT_FILE)
print("Saved:", PREDICTION_FILE)
print("Existing production model was not changed.")
