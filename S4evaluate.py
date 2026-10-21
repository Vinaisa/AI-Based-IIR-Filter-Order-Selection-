
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
    saved_data = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    raw_df = pd.read_csv(DATA_FILE)

    if "models" not in saved_data:
        raise KeyError("The saved model artifact has no 'models' entry.")

    if "training_columns" not in saved_data:
        raise KeyError(
            "The saved model artifact has no 'training_columns' entry."
        )

    # Recreate features using the actual training implementation.
    processed_df, X_all, _, _ = engineer_features(raw_df)

    training_columns = saved_data["training_columns"]
    X_all = X_all.reindex(columns=training_columns, fill_value=0)

    # Match the saved held-out specifications by spec_id and design method.
    key_columns = ["spec_id", "design_method"]

    if not all(col in test_df.columns for col in key_columns):
        raise ValueError("Test file is missing spec_id or design_method.")

    test_keys = test_df[key_columns].copy()
    test_keys["spec_id"] = test_keys["spec_id"].astype(str)
    processed_df = processed_df.copy()
    processed_df["spec_id"] = processed_df["spec_id"].astype(str)

    processed_keys = processed_df[key_columns].copy()
    processed_keys["spec_id"] = processed_keys["spec_id"].astype(str)

    lookup = processed_keys.reset_index().set_index(key_columns)["index"]
    indices = pd.MultiIndex.from_frame(test_keys)
    matched_indices = lookup.reindex(indices)

    if matched_indices.isna().any():
        raise ValueError(
            "Some test rows could not be matched to the original dataset."
        )

    X_test = X_all.loc[matched_indices.astype(int)].reset_index(drop=True)
    y_test = test_df["actual_order"].astype(int).to_numpy()

    if len(X_test) != len(y_test):
        raise ValueError("Test feature and target lengths do not match.")

    min_order = int(saved_data.get("minimum_order", 1))
    max_order = int(saved_data.get("maximum_order", 100))

    results = []

    for name, model in saved_data["models"].items():
        start = time.perf_counter()
        raw_predictions = model.predict(X_test)
        elapsed = time.perf_counter() - start

        predictions = np.rint(raw_predictions).astype(int)
        predictions = np.clip(predictions, min_order, max_order)

        results.append({
            "Model": name,
            "Exact-order accuracy (%)": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "Within ±1 order (%)": round(
                np.mean(np.abs(predictions - y_test) <= 1) * 100, 2
            ),
            "MAE": round(
                mean_absolute_error(y_test, predictions), 4
            ),
            "Prediction time (ms/sample)": round(
                elapsed * 1000 / len(X_test), 4
            )
        })

        # Compare with predictions saved by the original training script.
        if "predicted_order" in test_df.columns:
            saved_predictions = test_df["predicted_order"].astype(int).to_numpy()
            print(
                "\nMatches saved predictions:",
                name,
                np.mean(predictions == saved_predictions) * 100,
                "%"
            )

    results_df = pd.DataFrame(results).sort_values(
        by=["Exact-order accuracy (%)", "MAE"],
        ascending=[False, True]
    )

    print("\nSTAGE 4: VERIFIED MODEL EVALUATION")
    print("---------------------------------")
    print("Test samples:", len(X_test))
    print(results_df.to_string(index=False))

    results_df.to_csv(RESULT_FILE, index=False)
    print("\nSaved results to", RESULT_FILE)


if __name__ == "__main__":
    main()
