
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

from S3model import engineer_features

DATA = "iir_filter_dataset.csv"
RESULT_FILE = "response_model_comparison.csv"
PREDICTION_FILE = "response_model_predictions.csv"

RANDOM_STATE = 42

# Load dataset and create the same features used by S3model.py.
raw = pd.read_csv(DATA)
df, X, _, _ = engineer_features(raw)

y = df["reference_order"].astype(int)
groups = df["spec_id"]

# Use the same grouped 80/20 split as the earlier experiments.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]
train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

assert set(train_groups).isdisjoint(set(test_groups))

# Baseline: one shared model, tuned on training data only.
baseline_search = GridSearchCV(
    ExtraTreesRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    {
        "n_estimators": [700],
        "max_features": [0.7],
        "min_samples_leaf": [2],
    },
    scoring="neg_mean_absolute_error",
    cv=GroupKFold(n_splits=3),
    n_jobs=-1,
)
baseline_search.fit(X_train, y_train, groups=train_groups)
baseline_model = baseline_search.best_estimator_

baseline_prediction = np.clip(
    np.rint(baseline_model.predict(X_test)).astype(int),
    1,
    100,
)

# Train one model per response. Tune each only on that response's
# training rows; grouped validation keeps each spec_id together.
response_predictions = np.zeros(len(test_idx), dtype=int)
response_models = {}
response_searches = {}

for response in ["lowpass", "highpass"]:
    tr_mask = (
        train_df["filter_response"].to_numpy() == response
    )
    te_mask = (
        test_df["filter_response"].to_numpy() == response
    )

    X_response_train = X_train.loc[tr_mask]
    y_response_train = y_train.loc[tr_mask]
    groups_response_train = train_groups.loc[tr_mask]

    search = GridSearchCV(
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

    search.fit(
        X_response_train,
        y_response_train,
        groups=groups_response_train,
    )

    response_models[response] = search.best_estimator_
    response_searches[response] = search

    response_predictions[te_mask] = np.clip(
        np.rint(
            search.best_estimator_.predict(X_test.loc[te_mask])
        ).astype(int),
        1,
        100,
    )

    print(f"\n{response.upper()} model")
    print("Best parameters:", search.best_params_)
    print("Grouped CV MAE:", round(-search.best_score_, 4))

actual = y_test.to_numpy(dtype=int)

def metrics(name, prediction, mask=None):
    if mask is None:
        mask = np.ones(len(actual), dtype=bool)

    a = actual[mask]
    p = prediction[mask]
    errors = np.abs(a - p)

    return {
        "Model": name,
        "Samples": int(mask.sum()),
        "Exact accuracy (%)": round(
            np.mean(errors == 0) * 100, 2
        ),
        "Within ±1 (%)": round(
            np.mean(errors <= 1) * 100, 2
        ),
        "MAE": round(mean_absolute_error(a, p), 4),
    }

results = [
    metrics("Shared tuned Extra Trees", baseline_prediction),
    metrics("Separate low-pass/high-pass models", response_predictions),
]

results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_FILE, index=False)

print("\n--- OVERALL COMPARISON ---")
print(results_df.to_string(index=False))

print("\n--- PERFORMANCE BY RESPONSE ---")
for response in ["lowpass", "highpass"]:
    mask = (
        test_df["filter_response"].to_numpy() == response
    )

    print(f"\n{response.upper()}")
    print(metrics("Shared model", baseline_prediction, mask))
    print(metrics("Response-specific model", response_predictions, mask))

# Save test predictions for analysis.
output = test_df[
    [
        "spec_id",
        "filter_response",
        "filter_type",
        "design_method",
        "reference_order",
    ]
].copy()

output.rename(
    columns={"reference_order": "actual_order"},
    inplace=True,
)

output["shared_prediction"] = baseline_prediction
output["response_specific_prediction"] = response_predictions
output["shared_absolute_error"] = np.abs(
    output["actual_order"].to_numpy() - baseline_prediction
)
output["response_specific_absolute_error"] = np.abs(
    output["actual_order"].to_numpy() - response_predictions
)

output.to_csv(PREDICTION_FILE, index=False)

print("\nSaved:", RESULT_FILE)
print("Saved:", PREDICTION_FILE)
print("Production model was not changed.")
