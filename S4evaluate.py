
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, mean_absolute_error
from S3model import engineer_features

MODEL_FILE = "candidate_models.joblib"
TEST_FILE = "iir_test_data.csv"
DATA_FILE = "iir_filter_dataset.csv"
RESULT_FILE = "model_comparison.csv"


def main():
    print("Loading model, dataset and held-out test data...")

    saved_data = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    raw_df = pd.read_csv(DATA_FILE)

    if "models" not in saved_data:
        raise KeyError("Model artifact has no 'models' entry.")

    if "training_columns" not in saved_data:
        raise KeyError("Model artifact has no 'training_columns' entry.")

    models = saved_data["models"]
    training_columns = saved_data["training_columns"]

    if not models:
        raise ValueError("No trained models were found.")

    key_columns = ["spec_id", "design_method"]

    for column in key_columns:
        if column not in raw_df.columns or column not in test_df.columns:
            raise ValueError(
                f"Both CSV files must contain '{column}'."
            )

    if "reference_order" not in raw_df.columns:
        raise ValueError("Dataset is missing 'reference_order'.")

    target_column = (
        "actual_order"
        if "actual_order" in test_df.columns
        else "reference_order"
        if "reference_order" in test_df.columns
        else None
    )

    if target_column is None:
        raise ValueError(
            "Test CSV must contain actual_order or reference_order."
        )

    # Normalize matching keys.
    raw_df = raw_df.copy()
    test_df = test_df.copy()

    for df in (raw_df, test_df):
        df["spec_id"] = df["spec_id"].astype(str).str.strip()
        df["design_method"] = (
            df["design_method"].astype(str).str.strip()
        )

    if raw_df.duplicated(key_columns).any():
        raise ValueError(
            "Dataset contains duplicate spec_id/design_method keys."
        )

    if test_df.duplicated(key_columns).any():
        raise ValueError(
            "Test CSV contains duplicate spec_id/design_method keys."
        )

    # Match held-out test rows to the original dataset.
    indexed = raw_df.set_index(key_columns)
    test_index = pd.MultiIndex.from_frame(test_df[key_columns])

    missing_keys = ~test_index.isin(indexed.index)
    if missing_keys.any():
        raise ValueError(
            f"{int(missing_keys.sum())} test rows do not match the dataset."
        )

    matched_rows = indexed.reindex(test_index).reset_index(drop=True)

    # Verify test targets against the original reference orders.
    y_test = test_df[target_column].astype(int).to_numpy()
    original_targets = matched_rows["reference_order"].astype(int).to_numpy()

    if not np.array_equal(y_test, original_targets):
        raise ValueError(
            "Test targets do not match reference_order in the dataset. "
            "Restore the original held-out test CSV before evaluating."
        )

    # engineer_features returns the encoded feature DataFrame.
    X_all = engineer_features(raw_df)

    if not isinstance(X_all, pd.DataFrame):
        raise TypeError(
            "engineer_features() must return a pandas DataFrame."
        )

    if len(X_all) != len(raw_df):
        raise ValueError(
            "Feature row count does not match the original dataset."
        )

    # Keep row order aligned with raw_df.
    X_all = X_all.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Resolve held-out row positions using the verified keys.
    key_to_position = {
        (row.spec_id, row.design_method): position
        for position, row in enumerate(
            raw_df[key_columns].itertuples(index=False)
        )
    }

    positions = [
        key_to_position[(row.spec_id, row.design_method)]
        for row in test_df[key_columns].itertuples(index=False)
    ]

    X_test = X_all.iloc[positions].reset_index(drop=True)

    if len(X_test) != len(y_test) or len(y_test) == 0:
        raise ValueError("Invalid held-out test feature/target lengths.")

    minimum_order = int(saved_data.get("minimum_order", 1))
    maximum_order = int(saved_data.get("maximum_order", 100))

    results = []

    print("Evaluating saved models on verified test rows...")

    for name, model in models.items():
        start = time.perf_counter()
        raw_predictions = model.predict(X_test)
        elapsed = time.perf_counter() - start

        predictions = np.clip(
            np.rint(raw_predictions).astype(int),
            minimum_order,
            maximum_order
        )

        exact = accuracy_score(y_test, predictions) * 100
        within_one = (
            np.mean(np.abs(predictions - y_test) <= 1) * 100
        )
        mae = mean_absolute_error(y_test, predictions)

        results.append({
            "Model": name,
            "Exact-order accuracy (%)": round(exact, 2),
            "Within ±1 order (%)": round(within_one, 2),
            "MAE": round(mae, 4),
            "Prediction time (ms/sample)": round(
                elapsed * 1000 / len(X_test), 4
            )
        })

        # Compare predictions with those recorded by the training script.
        if "predicted_order" in test_df.columns:
            saved_predictions = (
                test_df["predicted_order"].astype(int).to_numpy()
            )

            match = np.mean(predictions == saved_predictions) * 100

            print(
                f"{name} — agreement with saved predictions: "
                f"{match:.2f}%"
            )

    results_df = pd.DataFrame(results).sort_values(
        by=["Exact-order accuracy (%)", "MAE"],
        ascending=[False, True]
    )

    print("\nSTAGE 4: HELD-OUT MODEL EVALUATION")
    print("---------------------------------")
    print("Verified test samples:", len(X_test))
    print(results_df.to_string(index=False))

    results_df.to_csv(RESULT_FILE, index=False)
    print(f"\nSaved results to {RESULT_FILE}")


if __name__ == "__main__":
    main()
