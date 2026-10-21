
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal


MODEL_FILE = "candidate_models.joblib"
MAX_ORDER = 100
FREQUENCY_POINTS = 8192


FAMILIES = {
    "1": ("Butterworth", "butter", signal.buttord),
    "2": ("Chebyshev I", "cheby1", signal.cheb1ord),
    "3": ("Chebyshev II", "cheby2", signal.cheb2ord),
}



def make_features(fs, fp, fsb, rp, rs, family, method, saved):
    response = "lowpass"

    row = {
        "sampling_hz": fs,
        "normalized_passband": fp / (fs / 2),
        "normalized_stopband": fsb / (fs / 2),
        "passband_ripple_db": rp,
        "stopband_attenuation_db": rs,
        "filter_response": response,
        "filter_type": family,
        "design_method": method,
    }

    # Match the training pipeline in S3model.py.
    transition_width_hz = abs(fsb - fp)
    normalized_transition_width = abs(
        row["normalized_stopband"] - row["normalized_passband"]
    )

    lower_edge = min(fp, fsb)
    upper_edge = max(fp, fsb)
    transition_ratio = upper_edge / max(lower_edge, 1e-12)

    row["normalized_transition_width"] = normalized_transition_width
    row["transition_ratio"] = transition_ratio
    row["transition_over_passband"] = (
        transition_width_hz / max(abs(fp), 1e-12)
    )
    row["ripple_attenuation_ratio"] = (
        rs / max(rp, 1e-12)
    )
    row["log_transition_ratio"] = np.log(
        max(transition_ratio, 1e-12)
    )
    row["log_attenuation"] = np.log1p(max(rs, 0))
    row["log_ripple"] = np.log1p(max(rp, 0))
    row["log_sampling_hz"] = np.log(max(fs, 1e-12))
    row["log_transition_width_hz"] = np.log1p(
        max(transition_width_hz, 0)
    )
    row["edge_sum_normalized"] = (
        row["normalized_passband"] + row["normalized_stopband"]
    )
    row["edge_product_normalized"] = (
        row["normalized_passband"] * row["normalized_stopband"]
    )

    numeric = saved["numeric_features"]
    categorical = saved["categorical_features"]

    encoded = pd.get_dummies(
        pd.DataFrame([row])[numeric + categorical],
        columns=categorical,
        dtype=float,
    )

    encoded = encoded.reindex(
        columns=saved["training_columns"],
        fill_value=0,
    )

    return encoded

def design_filter(family, order, cutoff, fs, rp, rs, method):
    """Design a low-pass filter at the specified cutoff."""

    if method == "Direct digital":
        if family == "Butterworth":
            return signal.butter(
                order, cutoff, btype="lowpass", fs=fs, output="ba"
            )
        if family == "Chebyshev I":
            return signal.cheby1(
                order, rp, cutoff, btype="lowpass", fs=fs, output="ba"
            )
        return signal.cheby2(
            order, rs, cutoff, btype="lowpass", fs=fs, output="ba"
        )

    # BLT: prewarp the digital cutoff, design an analog prototype,
    # then convert the prototype into a digital filter.
    warped = 2 * fs * np.tan(np.pi * cutoff / fs)

    if family == "Butterworth":
        b, a = signal.butter(
            order, warped, btype="lowpass", analog=True, output="ba"
        )
    elif family == "Chebyshev I":
        b, a = signal.cheby1(
            order, rp, warped, btype="lowpass",
            analog=True, output="ba"
        )
    else:
        b, a = signal.cheby2(
            order, rs, warped, btype="lowpass",
            analog=True, output="ba"
        )

    return signal.bilinear(b, a, fs=fs)


def measure_response(b, a, fs, fp, fstop):
    f, h = signal.freqz(
        b, a, worN=FREQUENCY_POINTS, fs=fs
    )
    db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    passband = f <= fp
    stopband = f >= fstop

    ripple = float(np.max(db[passband]) - np.min(db[passband]))
    attenuation = float(-np.max(db[stopband]))

    return f, db, ripple, attenuation


def find_valid_design(family, predicted, fs, fp, fstop, rp, rs, method):
    """
    Start with the AI-predicted order. Search cutoff frequencies for
    a design satisfying both requirements. Increase order if necessary.
    """

    for order in range(max(1, predicted), MAX_ORDER + 1):
        # Search the feasible low-pass cutoff interval.
        cutoffs = np.linspace(fp, fstop, 301)

        for cutoff in cutoffs:
            try:
                b, a = design_filter(
                    family, order, cutoff, fs, rp, rs, method
                )

                f, db, ripple, attenuation = measure_response(
                    b, a, fs, fp, fstop
                )

                if (
                    np.isfinite(ripple)
                    and np.isfinite(attenuation)
                    and ripple <= rp + 1e-6
                    and attenuation >= rs - 1e-6
                ):
                    return {
                        "order": order,
                        "cutoff": float(cutoff),
                        "b": b,
                        "a": a,
                        "f": f,
                        "db": db,
                        "ripple": ripple,
                        "attenuation": attenuation,
                        "passed": True,
                    }

            except (ValueError, FloatingPointError):
                continue

    return {"passed": False}


def run_family(family, model, saved, fs, fp, fstop, rp, rs, method):
    X = make_features(
        fs, fp, fstop, rp, rs, family, method, saved
    )
    predicted = int(np.clip(
        np.rint(model.predict(X)[0]), 1, MAX_ORDER
    ))

    print("\n" + "=" * 56)
    print(f"FILTER FAMILY: {family}")
    print("AI-predicted order:", predicted)
    print("Design method:", method)

    result = find_valid_design(
        family, predicted, fs, fp, fstop, rp, rs, method
    )

    if not result["passed"]:
        print(f"FAIL: no passing design found through order {MAX_ORDER}.")
        print("The AI prediction remains", predicted)
        return

    print("Final validated order:", result["order"])
    print("Chosen cutoff (Hz):", round(result["cutoff"], 4))
    print(
        f"Measured passband ripple: {result['ripple']:.4f} dB "
        f"(limit {rp:.4f} dB)"
    )
    print(
        f"Measured stopband attenuation: {result['attenuation']:.4f} dB "
        f"(minimum {rs:.4f} dB)"
    )
    print("Passband check: PASS")
    print("Stopband check: PASS")
    print("Overall result: SPECIFICATIONS MET")

    print("\nNumerator coefficients (b):")
    print(result["b"])
    print("\nDenominator coefficients (a):")
    print(result["a"])

    plt.figure(figsize=(10, 5))
    plt.plot(result["f"], result["db"], label="Designed response")
    plt.axvline(fp, linestyle="--", label="Passband edge")
    plt.axvline(fstop, linestyle="--", label="Stopband edge")
    plt.axhline(-rp, linestyle=":", label="Ripple reference")
    plt.axhline(-rs, linestyle=":", label="Attenuation reference")
    plt.title(
        f"{family} | AI order {predicted} | Validated order {result['order']}"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 56)
    print("STAGE 5: IIR FILTER DESIGN AND VALIDATION")
    print("=" * 56)

    saved = joblib.load(MODEL_FILE)
    model = saved["models"][saved["best_model_name"]]

    fs_input = float(input("Sampling frequency: "))
    unit = input("Frequency unit (Hz/kHz): ").strip().lower()

    if unit not in ("hz", "khz"):
        raise ValueError("Enter Hz or kHz.")

    scale = 1000.0 if unit == "khz" else 1.0
    fs = fs_input * scale

    fp = float(input("Passband edge frequency: ")) * scale
    fstop = float(input("Stopband edge frequency: ")) * scale
    rp = float(input("Maximum passband ripple (dB): "))
    rs = float(input("Minimum stopband attenuation (dB): "))

    if fs <= 0 or not (0 < fp < fstop < fs / 2):
        raise ValueError(
            "Require 0 < passband edge < stopband edge < Nyquist frequency."
        )
    if rp <= 0 or rs <= rp:
        raise ValueError(
            "Ripple must be positive and stopband attenuation greater than ripple."
        )

    print("\nFilter family:")
    print("1. Butterworth")
    print("2. Chebyshev I")
    print("3. Chebyshev II")
    print("4. Test all three")

    choice = input("Choose (1-4): ").strip()
    if choice not in ("1", "2", "3", "4"):
        raise ValueError("Choose 1, 2, 3, or 4.")

    print("\nDesign method:")
    print("1. Direct digital")
    print("2. BLT")

    method_choice = input("Choose (1-2): ").strip()
    if method_choice not in ("1", "2"):
        raise ValueError("Choose 1 or 2.")

    method = "Direct digital" if method_choice == "1" else "BLT"

    choices = ["1", "2", "3"] if choice == "4" else [choice]

    for family_choice in choices:
        family = FAMILIES[family_choice][0]
        run_family(
            family, model, saved, fs, fp, fstop, rp, rs, method
        )

    print("\nStage 5 test completed.")


if __name__ == "__main__":
    main()
