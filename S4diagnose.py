
import pandas as pd

df = pd.read_csv("iir_filter_dataset.csv")

print("DATASET DIAGNOSIS")
print("-----------------")

# 1. Dataset size and integrity
print("\n1. Dataset size:")
print("Rows:", len(df))
print("Unique specifications:", df["spec_id"].nunique())
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

# 2. Unique reference orders
print("\n2. Unique reference orders:")
print(sorted(df["reference_order"].unique()))

# 3. Order statistics by filter family
print("\n3. Order statistics by filter type:")
print(
    df.groupby("filter_type")["reference_order"]
    .agg(["min", "median", "mean", "max"])
    .round(2)
)

# 4. Order statistics by design method
print("\n4. Order statistics by design method:")
print(
    df.groupby("design_method")["reference_order"]
    .agg(["min", "median", "mean", "max"])
    .round(2)
)

# 5. Order statistics by response
if "filter_response" in df.columns:
    print("\n5. Order statistics by filter response:")
    print(
        df.groupby("filter_response")["reference_order"]
        .agg(["min", "median", "mean", "max"])
        .round(2)
    )

# 6. Verify paired design methods
print("\n6. Paired method-order check:")

paired = df.pivot_table(
    index=["spec_id", "filter_type", "filter_response"],
    columns="design_method",
    values="reference_order",
    aggfunc="first"
)

print("Paired specifications:", len(paired))
print("Incomplete pairs:", paired.isnull().any(axis=1).sum())

complete = paired.dropna()
print(
    "Mismatched paired orders:",
    (complete.nunique(axis=1) > 1).sum()
)

# 7. Correlation with reference order
print("\n7. Correlation with reference order:")

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

numeric_columns = [
    col for col in numeric_columns if col in df.columns
]

correlations = (
    df[numeric_columns]
    .corr(numeric_only=True)["reference_order"]
    .sort_values()
)

print(correlations.round(3))
