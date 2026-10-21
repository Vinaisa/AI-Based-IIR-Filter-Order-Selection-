
import numpy as np
import pandas as pd
from scipy import signal

# ============================================================
# IIR FILTER ORDER DATASET GENERATOR
# Responses: Low-pass, High-pass
# Families: Butterworth, Chebyshev I, Chebyshev II
# Methods: Direct digital, BLT
# ============================================================

OUTPUT_FILE = "iir_filter_dataset.csv"
TARGET_PER_BIN = 100
MAX_ATTEMPTS = 1_000_000
MAX_ORDER = 100
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

FILTERS = {
    "Butterworth": (signal.buttord, signal.butter),
    "Chebyshev I": (signal.cheb1ord, signal.cheby1),
    "Chebyshev II": (signal.cheb2ord, signal.cheby2),
}

RESPONSES = ["lowpass", "highpass"]
METHODS = ["Direct digital", "BLT"]

ORDER_BINS = [
    (1, 5),
    (6, 10),
    (11, 20),
    (21, 40),
    (41, 100),
]

# Balance each response, family and order range separately.
buckets = {
    (response, family, low, high): []
    for response in RESPONSES
    for family in FILTERS
    for low, high in ORDER_BINS
}

seen_specs = set()


def generate_spec(response):
    """Generate a valid low-pass or high-pass specification."""

    fs = float(np.exp(rng.uniform(np.log(250), np.log(96000))))

    edge_ratio = rng.uniform(0.04, 0.46)
    transition_ratio = np.exp(
        rng.uniform(np.log(1.005), np.log(2.4))
    )

    if response == "lowpass":
        fp_ratio = edge_ratio
        fsb_ratio = min(fp_ratio * transition_ratio, 0.98)

        if fsb_ratio <= fp_ratio:
            return None

    else:
        fp_ratio = edge_ratio
        fsb_ratio = fp_ratio / transition_ratio

        if fsb_ratio <= 0.005:
            return None

    fp = fp_ratio * fs / 2
    fsb = fsb_ratio * fs / 2

    rp = float(rng.uniform(0.05, 1.5))
    rs = float(rng.uniform(max(20.0, rp + 10.0), 100.0))

    if response == "lowpass":
        valid = 0 < fp < fsb < fs / 2
    else:
        valid = 0 < fsb < fp < fs / 2

    if not valid:
        return None

    key = (
        response,
        round(fs, 8),
        round(fp, 8),
        round(fsb, 8),
        round(rp, 6),
        round(rs, 6),
    )

    if key in seen_specs:
        return None

    return fs, fp, fsb, rp, rs, key


def calculate_orders(response, fs, fp, fsb, rp, rs):
    """Calculate reference orders using SciPy."""

    results = {}

    if response == "lowpass":
        wp, ws = fp, fsb
    else:
        wp, ws = fp, fsb

    for family, (order_function, _) in FILTERS.items():
        try:
            # Direct digital design using physical frequencies.
            digital_order, _ = order_function(
                wp, ws, rp, rs, fs=fs
            )

            # Prewarp the edges for analog design via BLT.
            warped_wp = 2 * fs * np.tan(np.pi * wp / fs)
            warped_ws = 2 * fs * np.tan(np.pi * ws / fs)

            blt_order, _ = order_function(
                warped_wp,
                warped_ws,
                rp,
                rs,
                analog=True,
            )

            digital_order = int(digital_order)
            blt_order = int(blt_order)

            # Equivalent specifications should give equal orders.
            if digital_order != blt_order:
                continue

            if not (1 <= digital_order <= MAX_ORDER):
                continue

            results[family] = digital_order

        except (ValueError, OverflowError, FloatingPointError):
            continue

    return results


def add_rows(
    spec_id, response, family, order,
    fs, fp, fsb, rp, rs
):
    """Create paired records for one specification."""

    common = {
        "spec_id": spec_id,
        "sampling_hz": fs,
        "passband_edge": fp,
        "stopband_edge": fsb,
        "normalized_passband": fp / (fs / 2),
        "normalized_stopband": fsb / (fs / 2),
        "passband_ripple_db": rp,
        "stopband_attenuation_db": rs,
        "filter_type": family,
        "filter_response": response,
        "reference_order": order,
    }

    rows = []

    for method in METHODS:
        row = common.copy()
        row["design_method"] = method
        rows.append(row)

    return rows


# ============================================================
# Generate the dataset
# ============================================================

dataset_rows = []
next_spec_id = 1

for attempt in range(1, MAX_ATTEMPTS + 1):

    if all(
        len(bucket) >= TARGET_PER_BIN
        for bucket in buckets.values()
    ):
        break

    response = RESPONSES[
        rng.integers(0, len(RESPONSES))
    ]

    spec = generate_spec(response)

    if spec is None:
        continue

    fs, fp, fsb, rp, rs, key = spec

    orders = calculate_orders(
        response, fs, fp, fsb, rp, rs
    )

    if not orders:
        continue

    candidates = []

    for family, order in orders.items():
        for low, high in ORDER_BINS:
            bucket = buckets[
                (response, family, low, high)
            ]

            if (
                low <= order <= high
                and len(bucket) < TARGET_PER_BIN
            ):
                candidates.append(
                    (family, low, high, order)
                )

    if not candidates:
        continue

    # Fill the most underfilled eligible bucket first.
    candidates.sort(
        key=lambda item: len(
            buckets[(response, item[0], item[1], item[2])]
        )
    )

    family, low, high, order = candidates[0]

    seen_specs.add(key)

    buckets[(response, family, low, high)].append(order)

    dataset_rows.extend(
        add_rows(
            next_spec_id,
            response,
            family,
            order,
            fs,
            fp,
            fsb,
            rp,
            rs,
        )
    )

    next_spec_id += 1

    if attempt % 25_000 == 0:
        filled = sum(len(v) for v in buckets.values())
        total = len(buckets) * TARGET_PER_BIN

        print(
            f"Attempts: {attempt:,} | "
            f"Buckets filled: {filled}/{total} | "
            f"Specifications: {next_spec_id - 1:,}"
        )


# ============================================================
# Save and verify
# ============================================================

if not dataset_rows:
    raise RuntimeError(
        "No valid samples generated. Check your SciPy installation."
    )

df = pd.DataFrame(dataset_rows)

columns = [
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
    "reference_order",
]

df = df[columns]
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset saved to:", OUTPUT_FILE)
print("Total rows:", len(df))
print("Unique specifications:", df["spec_id"].nunique())
print("Responses:", sorted(df["filter_response"].unique()))
print("Maximum reference order:", df["reference_order"].max())

print("\nRows by response, family and method:")
print(
    df.groupby(
        ["filter_response", "filter_type", "design_method"]
    ).size().to_string()
)

print("\nOrder-bin balance:")

incomplete = []

for response in RESPONSES:
    for family in FILTERS:
        print(f"\n{response} — {family}")

        for low, high in ORDER_BINS:
            count = len(
                buckets[(response, family, low, high)]
            )

            status = (
                "OK" if count == TARGET_PER_BIN
                else "INCOMPLETE"
            )

            print(
                f"  Orders {low:>2}-{high:<3}: "
                f"{count:>4}/{TARGET_PER_BIN} [{status}]"
            )

            if count < TARGET_PER_BIN:
                incomplete.append(
                    (response, family, low, high, count)
                )

print("\nRows per response:")
print(df.groupby("filter_response").size().to_string())

# Verify each specification has exactly two method records.
records_per_spec = df.groupby("spec_id").size()

if not (records_per_spec == 2).all():
    raise RuntimeError(
        "Dataset integrity check failed: each spec_id "
        "must have exactly two records."
    )

# Verify paired records share identical target labels.
orders_per_spec = df.groupby("spec_id")["reference_order"].nunique()

if not (orders_per_spec == 1).all():
    raise RuntimeError(
        "Dataset integrity check failed: reference orders differ "
        "within a specification group."
    )

print("\nIntegrity checks passed.")

if incomplete:
    print(
        "\nWARNING: Some order bins are incomplete. "
        "Review the counts before training."
    )
    print(
        "If needed, increase MAX_ATTEMPTS or adjust the "
        "specification-generation ranges."
    )
else:
    print("\nAll response/family/order bins reached their targets.")
