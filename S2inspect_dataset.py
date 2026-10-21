import numpy as np
import pandas as pd
from pathlib import Path

# Locate the dataset in the project folder.
ROOT = Path(__file__).resolve().parent
DATASET_FILE = ROOT / "iir_filter_dataset.csv"

EXPECTED_COLUMNS = [
    "spec_id",
    "sampling_hz",
    "passband_edge",
    "stopband_edge",
    "normalized_passband",
    "normalized_stopband",
    "passband_ripple_db",
    "stopband_attenuation_db",
    "filter_type",
    "design_method",
    "reference_order"
]

EXPECTED_FILTERS = {
    "Butterworth",
    "Chebyshev I",
    "Chebyshev II"
}

EXPECTED_METHODS = {
    "Direct digital",
    "Bilinear transform"
}


def inspect_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}\n"
            "Run generate_dataset.py first."
        )

    df = pd.read_csv(DATASET_FILE)

    print("=" * 60)
    print("IIR FILTER DATASET INSPECTION")
    print("=" * 60)

    # 1. Dataset dimensions.
    print("\n1. DATASET SIZE")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # 2. Check expected columns.
    print("\n2. COLUMN CHECK")
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)

    if not missing_columns:
        print("PASS: All expected columns are present.")
    else:
        print("FAIL: Missing columns:", sorted(missing_columns))

    print("Extra columns:", sorted(extra_columns) if extra_columns else "None")

    # 3. Data types.
    print("\n3. DATA TYPES")
    print(df.dtypes.to_string())

    # 4. Missing values.
    print("\n4. MISSING VALUES")
    missing = df.isnull().sum()
    print(missing.to_string())

    if missing.sum() == 0:
        print("PASS: No missing values.")
    else:
        print("WARNING: Missing values detected.")

    # 5. Duplicate rows.
    print("\n5. DUPLICATE ROWS")
    duplicate_count = int(df.duplicated().sum())
    print(f"Exact duplicate rows: {duplicate_count}")

    # 6. Filter families and design methods.
    print("\n6. FILTER FAMILIES")
    print(df["filter_type"].value_counts().to_string())

    print("\n7. DESIGN METHODS")
    print(df["design_method"].value_counts().to_string())

    unknown_filters = set(df["filter_type"].unique()) - EXPECTED_FILTERS
    unknown_methods = set(df["design_method"].unique()) - EXPECTED_METHODS

    print("Unexpected filter families:", unknown_filters or "None")
    print("Unexpected design methods:", unknown_methods or "None")

    # 7. Specification groups and combinations.
    print("\n8. SPECIFICATION GROUPS")
    print(f"Unique spec_id values: {df['spec_id'].nunique()}")

    combinations = df.groupby("spec_id").size()
    print("\nRows per specification group:")
    print(combinations.value_counts().sort_index().to_string())

    expected_pairs = {
        (family, method)
        for family in EXPECTED_FILTERS
        for method in EXPECTED_METHODS
    }

    incomplete_groups = []

    for spec_id, group in df.groupby("spec_id"):
        actual_pairs = set(
            zip(group["filter_type"], group["design_method"])
        )

        if actual_pairs != expected_pairs:
            incomplete_groups.append(spec_id)

    print(
        "Groups missing one or more combinations:",
        len(incomplete_groups)
    )
    # 8. Numerical validity.
    print("\n9. NUMERICAL VALIDITY")

    numeric_columns = [
        "sampling_hz",
        "passband_edge",
        "stopband_edge",
        "normalized_passband",
        "normalized_stopband",
        "passband_ripple_db",
        "stopband_attenuation_db",
        "reference_order"
    ]

    numeric_values = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Check for missing, infinite, or non-numeric values.
    finite_values = (
        numeric_values.notna().all().all()
        and np.isfinite(numeric_values.to_numpy()).all()
    )

    fs = numeric_values["sampling_hz"]
    fp = numeric_values["passband_edge"]
    fstop = numeric_values["stopband_edge"]
    rp = numeric_values["passband_ripple_db"]
    rs = numeric_values["stopband_attenuation_db"]
    order = numeric_values["reference_order"]

    valid_frequency = (
        (fs > 0)
        & (fp > 0)
        & (fp < fstop)
        & (fstop < fs / 2)
    )

    valid_ripple = (rp > 0) & (rs > rp)

    valid_order = (
        (order >= 1)
        & (order <= 100)
        & (order % 1 == 0)
    )

    print("All numeric values finite:", finite_values)
    print(
        "Invalid frequency specifications:",
        int((~valid_frequency).sum())
    )
    print(
        "Invalid ripple/attenuation specifications:",
        int((~valid_ripple).sum())
    )
    print("Invalid reference orders:", int((~valid_order).sum()))
    
    # 9. Order distribution.
    print("\n10. REFERENCE ORDER STATISTICS")
    print(df["reference_order"].describe().to_string())
    print("Minimum order:", df["reference_order"].min())
    print("Maximum order:", df["reference_order"].max())

    # 10. Normalized frequency consistency.
    print("\n11. NORMALIZED FREQUENCY CHECK")

    expected_wp = fp / (fs / 2)
    expected_ws = fstop / (fs / 2)

    wp_matches = (
        numeric_values["normalized_passband"]
        .sub(expected_wp).abs() < 1e-9
    )
    ws_matches = (
        numeric_values["normalized_stopband"]
        .sub(expected_ws).abs() < 1e-9
    )

    print("Passband normalization mismatches:", int((~wp_matches).sum()))
    print("Stopband normalization mismatches:", int((~ws_matches).sum()))

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
