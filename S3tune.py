
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error

from S3model import engineer_features

DATA = "iir_filter_dataset.csv"
MODEL_FILE = "candidate_models.joblib"
OUTPUT = "tuning_comparison.csv"
TEST_OUTPUT = "tuning_test_predictions.csv"

# Load and prepare the same dataset and feature representation.
raw = pd.read_csv(DATA)
df, X, numeric_features, categorical_features = engineer_features(raw)
y = df["reference_order"].astype(int)
groups = df["spec_id"]

# Preserve the same held-out specification split used by S3model.py.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train = groups.iloc[train_idx]

# Confirm the held-out test specifications are never used in tuning.
assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[test_idx])
)

cv = GroupKFold(n_splits=3)

models_to_tune = [
    (
        "Tuned Extra Trees",
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [400, 700],
            "max_features": [0.7, 1.0],
            "min_samples_leaf": [1, 2, 3],
            "max_depth": [None, 25, 40],
        },
    ),
    (
        "Tuned Random Forest",
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [400, 700],
            "max_features": [0.7, 1.0],
            "min_samples_leaf": [1, 2, 3],
            "max_depth": [None, 25, 40],
        },
    ),
]

results = []
fitted = {}

def evaluate(name, model, X_eval, y_eval):
    pred = np.clip(
        np.rint(model.predict(X_eval)).astype(int),
        1, 100
    )
    actual = np.asarray(y_eval, dtype=int)
    return {
        "Model": name,
        "Exact accuracy (%)": round(
            np.mean(pred == actual) * 100, 2
        ),
        "Within ±1 (%)": round(
            np.mean(np.abs(pred - actual) <= 1) * 100, 2
        ),
        "MAE": round(
            mean_absolute_error(actual, pred), 4
        ),
    }, pred

print("Tuning on training specifications only...\n")

for name, estimator, params in models_to_tune:
    search = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train, groups=g_train)

    fitted[name] = search.best_estimator_
    row, _ = evaluate(
        name, search.best_estimator_, X_test, y_test
    )
    row["CV MAE"] = round(-search.best_score_, 4)
    row["Best parameters"] = str(search.best_params_)
    results.append(row)

    print(name)
    print("  Best CV MAE:", round(-search.best_score_, 4))
    print("  Parameters:", search.best_params_)
    print("  Test:", row)

# Compare one shared model against separate models per filter family.
for family in sorted(df["filter_type"].unique()):
    tr_mask = (
        df.iloc[train_idx]["filter_type"].to_numpy() == family
    )
    te_mask = (
        df.iloc[test_idx]["filter_type"].to_numpy() == family
    )

    family_model = ExtraTreesRegressor(
        n_estimators=700,
        max_features=1.0,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    family_model.fit(X_train.loc[tr_mask], y_train.loc[tr_mask])

    row, _ = evaluate(
        f"Family-specific: {family}",
        family_model,
        X_test.loc[te_mask],
        y_test.loc[te_mask],
    )
    row["CV MAE"] = np.nan
    row["Best parameters"] = "Extra Trees, default baseline settings"
    results.append(row)
    fitted[f"Family-specific: {family}"] = family_model
    print("\n", row)

# Evaluate the currently saved model for a direct comparison.
try:
    artifact = joblib.load(MODEL_FILE)
    baseline_name = artifact["best_model_name"]
    baseline_model = artifact["models"][baseline_name]

    # Align test features with the saved model's training columns.
    saved_columns = artifact["training_columns"]
    X_aligned = X.reindex(columns=saved_columns, fill_value=0.0)

    row, _ = evaluate(
        f"Existing baseline: {baseline_name}",
        baseline_model,
        X_aligned.iloc[test_idx],
        y_test,
    )
    row["CV MAE"] = np.nan
    row["Best parameters"] = "Previously saved model"
    results.append(row)
    print("\n", row)
except Exception as exc:
    print("\nBaseline comparison unavailable:", exc)

results_df = pd.DataFrame(results).sort_values(
    ["Exact accuracy (%)", "MAE"],
    ascending=[False, True],
)
results_df.to_csv(OUTPUT, index=False)

print("\n--- FINAL COMPARISON ---")
print(results_df.to_string(index=False))
print("\nSaved comparison:", OUTPUT)

# Save predictions from the best tested model.
best_name = results_df.iloc[0]["Model"]
best_model = fitted.get(best_name)

if best_model is not None:
    _, predictions = evaluate(
        best_name, best_model, X_test, y_test
    )
    out = df.iloc[test_idx][
        ["spec_id", "filter_response", "filter_type",
         "design_method", "reference_order"]
    ].copy()
    out.rename(
        columns={"reference_order": "actual_order"},
        inplace=True,
    )
    out["predicted_order"] = predictions
    out["absolute_error"] = np.abs(
        out["actual_order"].to_numpy() - predictions
    )
    out.to_csv(TEST_OUTPUT, index=False)
    print("Saved predictions:", TEST_OUTPUT)
    print("Best tested candidate:", best_name)
    print("Note: this script does not overwrite the production model.")
