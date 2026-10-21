
import os
import shutil
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

DATA = "iir_filter_dataset.csv"
MODEL_FILE = "candidate_models.joblib"
RESULT_FILE = "model_comparison.csv"
TEST_FILE = "iir_test_data.csv"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "sampling_hz",
    "normalized_passband",
    "normalized_stopband",
    "normalized_transition_width",
    "transition_ratio",
    "transition_over_passband",
    "ripple_attenuation_ratio",
    "log_transition_ratio",
    "log_attenuation",
    "log_ripple",
    "passband_ripple_db",
    "stopband_attenuation_db",
    "log_sampling_hz",
    "log_transition_width_hz",
    "edge_sum_normalized",
    "edge_product_normalized",
]

CATEGORICAL_FEATURES = [
    "filter_response",
    "filter_type",
    "design_method",
]

REQUIRED_COLUMNS = [
    "spec_id",
    "sampling_hz",
    "passband_edge",
    "stopband_edge",
    "normalized_passband",
    "normalized_stopband",
    "passband_ripple_db",
    "stopband_attenuation_db",
    "filter_response",
    "filter_type",
    "design_method",
    "reference_order",
]


def engineer_features(data):
    """Build features consistently with the Streamlit dashboard."""
    df = data.copy()

    df["filter_response"] = (
        df["filter_response"].astype(str).str.strip().str.lower()
    )
    df["filter_type"] = df["filter_type"].astype(str).str.strip()
    df["design_method"] = df["design_method"].astype(str).str.strip()

    df["transition_width_hz"] = (
        df["stopband_edge"] - df["passband_edge"]
    ).abs()

    df["normalized_transition_width"] = (
        df["normalized_stopband"] - df["normalized_passband"]
    ).abs()

    lower_edge = df[["passband_edge", "stopband_edge"]].min(axis=1)
    upper_edge = df[["passband_edge", "stopband_edge"]].max(axis=1)

    df["transition_ratio"] = (
        upper_edge / lower_edge.clip(lower=1e-12)
    )

    # Keep this definition identical to make_features() in the dashboard.
    df["transition_over_passband"] = (
        df["transition_width_hz"]
        / df["passband_edge"].abs().clip(lower=1e-12)
    )

    df["ripple_attenuation_ratio"] = (
        df["stopband_attenuation_db"]
        / df["passband_ripple_db"].clip(lower=1e-12)
    )

    df["log_transition_ratio"] = np.log(
        df["transition_ratio"].clip(lower=1e-8)
    )
    df["log_attenuation"] = np.log1p(
        df["stopband_attenuation_db"].clip(lower=0)
    )
    df["log_ripple"] = np.log1p(
        df["passband_ripple_db"].clip(lower=0)
    )
    df["log_sampling_hz"] = np.log(
        df["sampling_hz"].clip(lower=1e-12)
    )
    df["log_transition_width_hz"] = np.log1p(
        df["transition_width_hz"].clip(lower=0)
    )

    df["edge_sum_normalized"] = (
        df["normalized_passband"] + df["normalized_stopband"]
    )
    df["edge_product_normalized"] = (
        df["normalized_passband"] * df["normalized_stopband"]
    )

    encoded = pd.get_dummies(
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        columns=CATEGORICAL_FEATURES,
        dtype=float,
    )

    return encoded


def score_model(model, X, y):
    """Return rounded order predictions and evaluation metrics."""
    raw = model.predict(X)
    predictions = np.clip(
        np.rint(raw).astype(int),
        int(y.min()),
        int(y.max()),
    )

    actual = np.asarray(y, dtype=int)
    exact = float(np.mean(predictions == actual) * 100)
    within_one = float(
        np.mean(np.abs(predictions - actual) <= 1) * 100
    )
    mae = float(mean_absolute_error(actual, predictions))

    return predictions, exact, within_one, mae


def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            f"Cannot find {DATA}. Run this script from your project folder."
        )

    print("Loading dataset...")
    original = pd.read_csv(DATA)

    missing = sorted(set(REQUIRED_COLUMNS) - set(original.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    original = original.replace([np.inf, -np.inf], np.nan)
    original = original.dropna(subset=REQUIRED_COLUMNS).copy()

    if original.empty:
        raise ValueError("No valid dataset rows remain after cleaning.")

    if not original["filter_response"].str.lower().isin(
        ["lowpass", "highpass"]
    ).all():
        raise ValueError("Unexpected filter_response value.")

    if (original["reference_order"] < 1).any():
        raise ValueError("Reference orders must be positive.")

    if not np.allclose(
        original["reference_order"],
        np.rint(original["reference_order"]),
    ):
        raise ValueError("Reference orders must be integers.")

    original["reference_order"] = (
        original["reference_order"].astype(int)
    )

    X = engineer_features(original)
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().any().any():
        raise ValueError("Engineered features contain missing values.")

    y = original["reference_order"]
    groups = original["spec_id"]

    # Keep all rows sharing a spec_id together during evaluation.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Dataset rows: {len(original)}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Holdout rows: {len(test_idx)}")
    print("Training Extra Trees regressor...")

    model = ExtraTreesRegressor(
        n_estimators=700,
        max_features=0.7,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    test_predictions, exact, within_one, mae = score_model(
        model, X_test, y_test
    )
    prediction_ms = (
        time.perf_counter() - start
    ) * 1000 / len(X_test)

    print("\n--- HELD-OUT EVALUATION ---")
    print(f"Exact-order accuracy: {exact:.2f}%")
    print(f"Within ±1 order:      {within_one:.2f}%")
    print(f"MAE:                  {mae:.4f}")
    print(f"Training time:        {training_seconds:.2f} s")
    print(f"Prediction time:      {prediction_ms:.4f} ms/sample")

    results = pd.DataFrame([{
        "Model": "Extra Trees Regressor",
        "Exact-order accuracy (%)": round(exact, 2),
        "Within ±1 order (%)": round(within_one, 2),
        "MAE": round(mae, 4),
        "Training time (s)": round(training_seconds, 3),
        "Prediction time (ms/sample)": round(prediction_ms, 4),
    }])
    results.to_csv(RESULT_FILE, index=False)

    # Save original test specifications alongside predictions.
    test_output = original.iloc[test_idx][REQUIRED_COLUMNS].copy()
    test_output["predicted_order"] = test_predictions
    test_output["absolute_error"] = np.abs(
        test_output["reference_order"].to_numpy() - test_predictions
    )
    test_output.to_csv(TEST_FILE, index=False)

    # Evaluate low-pass and high-pass performance separately.
    print("\n--- RESPONSE-SPECIFIC ACCURACY ---")
    response_values = (
        original.iloc[test_idx]["filter_response"].to_numpy()
    )
    actual_test = y_test.to_numpy()

    for response in ["lowpass", "highpass"]:
        mask = response_values == response
        if mask.any():
            response_exact = np.mean(
                test_predictions[mask] == actual_test[mask]
            ) * 100
            response_mae = mean_absolute_error(
                actual_test[mask], test_predictions[mask]
            )
            print(
                f"{response}: exact={response_exact:.2f}%, "
                f"MAE={response_mae:.4f}, samples={mask.sum()}"
            )

    # Refit on all available data for the production artifact.
    print("\nRefitting model on all valid dataset rows...")
    production_model = ExtraTreesRegressor(
        n_estimators=700,
        max_features=0.7,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    production_model.fit(X, y)

    artifact = {
        "models": {"Extra Trees Regressor": production_model},
        "best_model_name": "Extra Trees Regressor",
        "features": list(X.columns),
        "numeric_features": NUMERIC_FEATURES.copy(),
        "categorical_features": CATEGORICAL_FEATURES.copy(),
        "input_columns": (
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ),
        "one_hot_encoded": True,
        "training_columns": list(X.columns),
        "minimum_order": int(y.min()),
        "maximum_order": int(y.max()),
        "supported_responses": sorted(
            original["filter_response"].unique().tolist()
        ),
        "supported_filter_types": sorted(
            original["filter_type"].unique().tolist()
        ),
        "supported_design_methods": sorted(
            original["design_method"].unique().tolist()
        ),
    }

    # Preserve the current model before replacing it.
    if os.path.exists(MODEL_FILE):
        backup = MODEL_FILE + ".before_s3model.bak"
        if not os.path.exists(backup):
            shutil.copy2(MODEL_FILE, backup)
            print(f"Previous model preserved at: {backup}")
        else:
            print(f"Existing backup preserved: {backup}")

    temp_file = MODEL_FILE + ".tmp"
    joblib.dump(artifact, temp_file)
    os.replace(temp_file, MODEL_FILE)

    # Check that the saved model's feature schema works.
    saved = joblib.load(MODEL_FILE)
    check_model = saved["models"][saved["best_model_name"]]
    check_prediction = check_model.predict(X.iloc[:2])

    if len(check_prediction) != min(2, len(X)):
        raise RuntimeError("Saved model smoke test failed.")

    print("\n--- COMPLETE ---")
    print(f"Production model saved: {MODEL_FILE}")
    print(f"Metrics saved:          {RESULT_FILE}")
    print(f"Holdout predictions:    {TEST_FILE}")
    print("Saved-model smoke test: PASS")
    print(
        "Note: holdout metrics measure the model before full-data refitting."
    )


if __name__ == "__main__":
    main()
