
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

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
    "filter_response",
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
    "BLT"
}

EXPECTED_RESPONSES = {
    "lowpass",
    "highpass"
}


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("IIR FILTER DATASET INSPECTION")
print("=" * 60)

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_FILE}"
    )

df = pd.read_csv(DATASET_FILE)


# ============================================================
# 1. DATASET SIZE
# ============================================================

print("\n1. DATASET SIZE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. COLUMN CHECK
# ============================================================

print("\n2. COLUMN CHECK")

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

extra_columns = [
    col for col in df.columns
    if col not in EXPECTED_COLUMNS
]

if not missing_columns:
    print("PASS: All expected columns are present.")
else:
    print("FAIL: Missing columns:", missing_columns)

if extra_columns:
    print("Unexpected extra columns:", extra_columns)
else:
    print("PASS: No unexpected extra columns.")

if missing_columns:
    raise ValueError(
        "Required columns are missing. "
        "Check the dataset before continuing."
    )


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n3. DATA TYPES")
print(df.dtypes)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n4. MISSING VALUES")

missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() == 0:
    print("PASS: No missing values.")
else:
    print("WARNING: Missing values detected.")


# ============================================================
# 5. DUPLICATE ROWS
# ============================================================

print("\n5. DUPLICATE ROWS")

duplicate_count = df.duplicated().sum()
print(f"Exact duplicate rows: {duplicate_count}")

if duplicate_count == 0:
    print("PASS: No exact duplicate rows.")
else:
    print("WARNING: Duplicate rows detected.")


# ============================================================
# 6. FILTER FAMILIES
# ============================================================

print("\n6. FILTER FAMILIES")

filter_counts = df["filter_type"].value_counts(dropna=False)
print(filter_counts)

actual_filters = set(df["filter_type"].dropna().unique())
unexpected_filters = actual_filters - EXPECTED_FILTERS

if not unexpected_filters:
    print("PASS: All filter families are recognized.")
else:
    print("Unexpected filter families:", unexpected_filters)


# ============================================================
# 7. DESIGN METHODS
# ============================================================

print("\n7. DESIGN METHODS")

method_counts = df["design_method"].value_counts(dropna=False)
print(method_counts)

actual_methods = set(df["design_method"].dropna().unique())
unexpected_methods = actual_methods - EXPECTED_METHODS

if not unexpected_methods:
    print("PASS: All design methods are recognized.")
else:
    print("Unexpected design methods:", unexpected_methods)


# ============================================================
# 8. SPECIFICATION GROUPS
# ============================================================

print("\n8. SPECIFICATION GROUPS")

print(f"Unique spec_id values: {df['spec_id'].nunique()}")

rows_per_group = df.groupby(
    "spec_id", dropna=False
).size()

print("\nRows per specification group:")
print(rows_per_group.value_counts().sort_index())

incomplete_groups = []

for spec_id, group in df.groupby("spec_id", dropna=False):
    actual_group_methods = set(group["design_method"])

    if (
        len(group) != 2
        or actual_group_methods != EXPECTED_METHODS
    ):
        incomplete_groups.append(spec_id)

print(
    "Groups missing one or more design methods:",
    len(incomplete_groups)
)

if len(incomplete_groups) == 0:
    print("PASS: Every specification group has both design methods.")
else:
    print("WARNING: Some specification groups are incomplete.")


# ============================================================
# 9. NUMERICAL VALIDITY
# ============================================================

print("\n9. NUMERICAL VALIDITY")

numeric_columns = [
    "spec_id",
    "sampling_hz",
    "passband_edge",
    "stopband_edge",
    "normalized_passband",
    "normalized_stopband",
    "passband_ripple_db",
    "stopband_attenuation_db",
    "reference_order"
]

numeric_data = df[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)

all_numeric_finite = np.isfinite(
    numeric_data.to_numpy(dtype=float)
).all()

print(f"All numeric values finite: {all_numeric_finite}")

fs = numeric_data["sampling_hz"]
fp = numeric_data["passband_edge"]
fstop = numeric_data["stopband_edge"]
rp = numeric_data["passband_ripple_db"]
rs = numeric_data["stopband_attenuation_db"]
order = numeric_data["reference_order"]

response = (
    df["filter_response"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.replace("-", "", regex=False)
    .str.replace("_", "", regex=False)
    .str.replace(" ", "", regex=False)
)

lowpass = response.isin(["lowpass", "lp"])
highpass = response.isin(["highpass", "hp"])

unknown_responses = ~(lowpass | highpass)

valid_frequency = (
    (fs > 0)
    & (fp > 0)
    & (fstop > 0)
    & (fp < fs / 2)
    & (fstop < fs / 2)
    & (
        (lowpass & (fp < fstop))
        | (highpass & (fstop < fp))
    )
)

invalid_frequency = (
    ~valid_frequency
    | unknown_responses
).fillna(True)

print(
    "Invalid frequency specifications:",
    int(invalid_frequency.sum())
)

invalid_ripple_attenuation = (
    (rp <= 0)
    | (rs <= 0)
    | (rs <= rp)
    | rp.isna()
    | rs.isna()
)

print(
    "Invalid ripple/attenuation specifications:",
    int(invalid_ripple_attenuation.sum())
)

invalid_orders = (
    (order < 1)
    | (order % 1 != 0)
    | order.isna()
)

print(
    "Invalid reference orders:",
    int(invalid_orders.sum())
)

print(
    "Unknown filter responses:",
    int(unknown_responses.sum())
)


# ============================================================
# 10. REFERENCE ORDER STATISTICS
# ============================================================

print("\n10. REFERENCE ORDER STATISTICS")

print(order.describe())

if order.notna().any():
    print(f"Minimum order: {order.min():g}")
    print(f"Maximum order: {order.max():g}")
else:
    print("No valid reference-order values available.")


# ============================================================
# 11. NORMALIZED FREQUENCY CHECK
# ============================================================

print("\n11. NORMALIZED FREQUENCY CHECK")

expected_normalized_passband = (
    numeric_data["passband_edge"]
    / (numeric_data["sampling_hz"] / 2)
)

expected_normalized_stopband = (
    numeric_data["stopband_edge"]
    / (numeric_data["sampling_hz"] / 2)
)

passband_mismatch = ~np.isclose(
    numeric_data["normalized_passband"],
    expected_normalized_passband,
    rtol=1e-5,
    atol=1e-8
)

stopband_mismatch = ~np.isclose(
    numeric_data["normalized_stopband"],
    expected_normalized_stopband,
    rtol=1e-5,
    atol=1e-8
)

print(
    "Passband normalization mismatches:",
    int(passband_mismatch.sum())
)

print(
    "Stopband normalization mismatches:",
    int(stopband_mismatch.sum())
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n12. FINAL VALIDATION SUMMARY")

checks = {
    "Required columns present": len(missing_columns) == 0,
    "No missing values": df.isnull().sum().sum() == 0,
    "No exact duplicate rows": duplicate_count == 0,
    "Known filter families": len(unexpected_filters) == 0,
    "Known design methods": len(unexpected_methods) == 0,
    "Complete specification groups": len(incomplete_groups) == 0,
    "Finite numeric values": all_numeric_finite,
    "Valid frequency specifications": invalid_frequency.sum() == 0,
    "Valid ripple and attenuation": invalid_ripple_attenuation.sum() == 0,
    "Valid reference orders": invalid_orders.sum() == 0,
    "Known filter responses": unknown_responses.sum() == 0,
    "Correct passband normalization": passband_mismatch.sum() == 0,
    "Correct stopband normalization": stopband_mismatch.sum() == 0
}

for check_name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {check_name}")

print("\n" + "=" * 60)

if all(checks.values()):
    print("INSPECTION COMPLETE: ALL CHECKS PASSED")
else:
    print("INSPECTION COMPLETE: SOME CHECKS FAILED")

print("=" * 60)
