import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import signal


st.set_page_config(
    page_title="AI-Based IIR Filter Design",
    page_icon=None,
    layout="wide",
)

MODEL_FILE = "candidate_models.joblib"
MAX_ORDER = 100
N_POINTS = 16384
CUTOFF_POINTS = 1001
TOL = 1e-5

FAMILIES = ["Butterworth", "Chebyshev I", "Chebyshev II"]
METHODS = ["Direct digital", "BLT"]


st.title("AI-Based IIR Filter Order Selection")
st.caption(
    "DSP Project | AI order prediction, conventional design comparison "
    "and measured response validation"
)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_FILE)
    model = saved["models"][saved["best_model_name"]]
    return saved, model


def make_features(fs, fp, fsb, rp, rs, family, method, saved, response="lowpass"):
    """Build exactly the feature columns used by S3model.py."""
    if response not in ("lowpass", "highpass"):
        raise ValueError("Response must be lowpass or highpass.")

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
    row["ripple_attenuation_ratio"] = rs / max(rp, 1e-12)
    row["log_transition_ratio"] = np.log(max(transition_ratio, 1e-12))
    row["log_attenuation"] = np.log1p(max(rs, 0))
    row["log_ripple"] = np.log1p(max(rp, 0))
    row["log_sampling_hz"] = np.log(max(fs, 1e-12))
    row["log_transition_width_hz"] = np.log1p(max(transition_width_hz, 0))
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
    return encoded.reindex(
        columns=saved["training_columns"],
        fill_value=0,
    )


def design_filter(
           family, order, cutoff, fs, rp, rs, method, response="lowpass"
):
    """Design a low-pass or high-pass IIR filter."""
    if response not in ("lowpass", "highpass"):
        raise ValueError("Response must be lowpass or highpass.")

    if method == "Direct digital":
        if family == "Butterworth":
            return signal.butter(order, cutoff, btype=response, fs=fs)
        if family == "Chebyshev I":
            return signal.cheby1(order, rp, cutoff, btype=response, fs=fs)
        return signal.cheby2(order, rs, cutoff, btype=response, fs=fs)

    # Prewarp the digital cutoff for the bilinear-transform implementation.
    warped = 2 * fs * np.tan(np.pi * cutoff / fs)
    if family == "Butterworth":
        b, a = signal.butter(order, warped, btype=response, analog=True)
    elif family == "Chebyshev I":
        b, a = signal.cheby1(order, rp, warped, btype=response, analog=True)
    else:
        b, a = signal.cheby2(order, rs, warped, btype=response, analog=True)
    return signal.bilinear(b, a, fs=fs)


def conventional_minimum_order(
    family, fs, fp, fsb, rp, rs, method, response="lowpass"
):
    """Calculate conventional minimum order and critical frequency."""
    if response not in ("lowpass", "highpass"):
        raise ValueError("Response must be lowpass or highpass.")

    # fp and fsb retain their semantic meanings in both response types:
    # fp is the passband edge and fsb is the stopband edge. For high-pass,
    # fp is numerically greater than fsb, as required by SciPy's order tools.
    if method == "Direct digital":
        if family == "Butterworth":
            order, cutoff = signal.buttord(fp, fsb, rp, rs, fs=fs)
        elif family == "Chebyshev I":
            order, cutoff = signal.cheb1ord(fp, fsb, rp, rs, fs=fs)
        else:
            order, cutoff = signal.cheb2ord(fp, fsb, rp, rs, fs=fs)
        return int(order), float(np.asarray(cutoff).reshape(-1)[0])
    
    wp = 2 * fs * np.tan(np.pi * fp / fs)
    ws = 2 * fs * np.tan(np.pi * fsb / fs)
    if family == "Butterworth":
        order, cutoff = signal.buttord(wp, ws, rp, rs, analog=True)
    elif family == "Chebyshev I":
        order, cutoff = signal.cheb1ord(wp, ws, rp, rs, analog=True)
    else:
        order, cutoff = signal.cheb2ord(wp, ws, rp, rs, analog=True)

    cutoff_analog = float(np.asarray(cutoff).reshape(-1)[0])
    cutoff_digital = fs / np.pi * np.arctan(cutoff_analog / (2 * fs))
    return int(order), float(cutoff_digital)


def measure_filter(b, a, fs, fp, fsb, response="lowpass"):
    freq, response_values = signal.freqz(b, a, worN=N_POINTS, fs=fs)
    magnitude = 20 * np.log10(np.maximum(np.abs(response_values), 1e-12))

    if response == "lowpass":
        passband = freq <= fp
        stopband = freq >= fsb
    elif response == "highpass":
        passband = freq >= fp
        stopband = freq <= fsb
    else:
        raise ValueError("Response must be lowpass or highpass.")

    if not np.any(passband) or not np.any(stopband):
        raise ValueError("The selected specification leaves an empty frequency band.")

    ripple = float(np.max(magnitude[passband]) - np.min(magnitude[passband]))
    attenuation = float(-np.max(magnitude[stopband]))
    return freq, magnitude, ripple, attenuation


def evaluate_design(
    family, order, cutoff, fs, fp, fsb, rp, rs, method, response="lowpass"
):
    try:
        b, a = design_filter(
            family, order, cutoff, fs, rp, rs, method, response
        )
        freq, magnitude, ripple, attenuation = measure_filter(
            b, a, fs, fp, fsb, response
        )
        if not (
            np.all(np.isfinite(b))
            and np.all(np.isfinite(a))
            and np.isfinite(ripple)
            and np.isfinite(attenuation)
            and np.all(np.isfinite(magnitude))
        ):
            return None
        return {
            "order": int(order),
            "cutoff": float(cutoff),
            "b": b,
            "a": a,
            "freq": freq,
            "mag": magnitude,
            "ripple": ripple,
            "attenuation": attenuation,
            "pass_ok": ripple <= rp + TOL,
            "stop_ok": attenuation >= rs - TOL,
        }
    except (ValueError, FloatingPointError, OverflowError):
        return None


def find_cutoff_for_fixed_order(
    family, order, fs, fp, fsb, rp, rs, method, response="lowpass"
):
    """Search cutoffs without changing order; return the best passing design."""
    if response == "lowpass":
        low_cutoff, high_cutoff = fp, fsb
    elif response == "highpass":
        low_cutoff, high_cutoff = fsb, fp
    else:
        raise ValueError("Response must be lowpass or highpass.")
    
    cutoffs = np.linspace(low_cutoff, high_cutoff, CUTOFF_POINTS)
    passing = []
    for cutoff in cutoffs:
        result = evaluate_design(
            family, order, cutoff, fs, fp, fsb, rp, rs, method, response
        )
        if result and result["pass_ok"] and result["stop_ok"]:
            score = (
                max(0.0, result["attenuation"] - rs),
                max(0.0, rp - result["ripple"]),
            )
            passing.append((score, result))

    if not passing:
        return None
    passing.sort(key=lambda item: item[0])
    return passing[0][1]


def build_result(
           family, order, cutoff, fs, fp, fsb, rp, rs, method, response="lowpass"
):
    return evaluate_design(
        family, order, cutoff, fs, fp, fsb, rp, rs, method, response
    )


try:
    saved, model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.info(
        "Keep candidate_models.joblib in the same folder as S6app(bp).py."
    )
    st.stop()


with st.sidebar:
    st.header("Filter specifications")

    fs_value = st.number_input(
        "Sampling frequency",
        min_value=1.0,
        value=500.0,
        step=10.0,
    )

    unit = st.selectbox("Frequency unit", ["Hz", "kHz"])
    scale = 1000.0 if unit == "kHz" else 1.0
    fs = fs_value * scale

    fp_value = st.number_input(
        "Passband edge",
        min_value=0.001,
        value=30.0,
        step=1.0,
    )

    fsb_value = st.number_input(
        "Stopband edge",
        min_value=0.002,
        value=60.0,
        step=1.0,
    )

    rp = st.number_input(
        "Maximum passband ripple (dB)",
        min_value=0.01,
        value=1.0,
        step=0.1,
    )

    rs = st.number_input(
        "Minimum stopband attenuation (dB)",
        min_value=0.1,
        value=40.0,
        step=1.0,
    )

    response_label = st.selectbox("Filter response", ["Low-pass", "High-pass"])
    response = "lowpass" if response_label == "Low-pass" else "highpass"
    
    family = st.selectbox("Filter family", FAMILIES)
    method = st.selectbox("Design method", METHODS)

    run = st.button(
        "Predict and compare designs",
        type="primary",
        use_container_width=True,
    )


st.subheader("How it works")
st.write(
    "The AI predicts an order. The application tests that exact order "
    "without silently increasing it, then independently calculates and "
    "validates a conventional minimum-order design."
)

st.caption(
    "Supports low-pass and high-pass responses. Frequency inputs may use Hz or kHz. "
    "Both designs are checked against the same specifications."
)


if run:
    fp = fp_value * scale
    fsb = fsb_value * scale

    valid_edges = (0 < fp < fsb < fs / 2) if response == "lowpass" else (0 < fsb < fp < fs / 2)
    if not valid_edges:
        if response == "lowpass":
            st.error("For a low-pass filter, require 0 < passband edge < stopband edge < Nyquist frequency.")
        else:
            st.error("For a high-pass filter, require 0 < stopband edge < passband edge < Nyquist frequency.")
        st.stop()

    if rp <= 0 or rs <= rp:
        st.error(
            "Passband ripple must be positive, and stopband attenuation "
            "must be greater than passband ripple."
        )
        st.stop()

    X = make_features(
        fs, fp, fsb, rp, rs, family, method, saved, response
    )

    predicted = int(np.clip(
        np.rint(model.predict(X)[0]), 1, MAX_ORDER
    ))

    with st.spinner("Calculating and validating both designs..."):
        try:
            theoretical_order, theoretical_cutoff = (
                conventional_minimum_order(
                    family, fs, fp, fsb, rp, rs, method, response
                )
            )
        except Exception as exc:
            st.error(
                f"Conventional minimum-order calculation failed: {exc}"
            )
            st.stop()

        ai_result = find_cutoff_for_fixed_order(
            family, predicted, fs, fp, fsb, rp, rs, method, response
        )

        conventional_result = build_result(
            family,
            theoretical_order,
            theoretical_cutoff,
            fs,
            fp,
            fsb,
            rp,
            rs,
            method,
            response,
        )

    st.divider()
    st.subheader("Order comparison")

    c1, c2, c3 = st.columns(3)
    c1.metric("AI-predicted order", predicted)
    c2.metric("Conventional minimum order", theoretical_order)

    difference = predicted - theoretical_order
    c3.metric("AI order difference", f"{difference:+d}")

    if difference == 0:
        st.success(
            "The AI prediction matches the conventional minimum order."
        )
    elif difference > 0:
        st.warning(
            f"The AI predicts {difference} additional order(s) compared "
            "with the conventional minimum-order calculation."
        )
    else:
        st.info(
            "The AI predicts a lower order than the conventional "
            "minimum-order result. Check whether that design can satisfy "
            "the specifications."
        )

    st.markdown("### AI-predicted order: exact-order validation")

    if ai_result is None:
        st.error(
            f"No cutoff found that satisfies both specifications at "
            f"the AI-predicted order {predicted}. The app has not silently "
            "increased the order."
        )
    else:
        a1, a2, a3 = st.columns(3)
        a1.metric("Order tested", ai_result["order"])
        a2.metric("Measured passband ripple", f'{ai_result["ripple"]:.4f} dB')
        a3.metric(
            "Measured stopband attenuation",
            f'{ai_result["attenuation"]:.4f} dB',
        )

        st.write(f"Selected cutoff frequency: **{ai_result['cutoff']:.4f} Hz**")

        if ai_result["pass_ok"] and ai_result["stop_ok"]:
            st.success(
                f"PASS: AI-predicted order {predicted} satisfies both "
                "specifications."
            )
        else:
            st.error(
                f"FAIL: AI-predicted order {predicted} does not satisfy "
                "both specifications."
            )

    st.markdown("### Conventional minimum-order validation")

    if conventional_result is None:
        st.error(
            "The conventional order function returned a design that "
            "could not be measured reliably."
        )
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Order tested", conventional_result["order"])
        t2.metric(
            "Measured passband ripple",
            f'{conventional_result["ripple"]:.4f} dB',
        )
        t3.metric(
            "Measured stopband attenuation",
            f'{conventional_result["attenuation"]:.4f} dB',
        )

        st.write(
            "Critical cutoff frequency: "
            f"**{conventional_result['cutoff']:.4f} Hz**"
        )

        if (
            conventional_result["pass_ok"]
            and conventional_result["stop_ok"]
        ):
            st.success(
                "PASS: The conventional minimum-order design satisfies "
                "both specifications."
            )
        else:
            st.error(
                "The conventional order function's design did not pass "
                "both measured checks. Inspect the implementation and "
                "frequency response before drawing conclusions."
            )

    st.markdown("### Frequency-response comparison")

    fig, ax = plt.subplots(figsize=(12, 6))

    if ai_result is not None:
        ax.plot(
            ai_result["freq"],
            ai_result["mag"],
            label=f"AI design (order {ai_result['order']})",
        )

    if conventional_result is not None:
        ax.plot(
            conventional_result["freq"],
            conventional_result["mag"],
            linestyle="--",
            label=(
                "Conventional design "
                f"(order {conventional_result['order']})"
            ),
        )

    ax.axvline(fp, linestyle=":", label="Passband edge")
    ax.axvline(fsb, linestyle=":", label="Stopband edge")
    ax.axhline(-rp, linestyle="-.", label="−Ripple limit")
    ax.axhline(-rs, linestyle="-.", label="−Attenuation requirement")

    ax.set_title(f"{family} {response_label} IIR Filter | {method}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, fs / 2)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Summary table")

    rows = []

    for label, result in [
        ("AI-predicted design", ai_result),
        ("Conventional minimum-order design", conventional_result),
    ]:
        if result is None:
            rows.append({
                "Design": label,
                "Order": None,
                "Cutoff (Hz)": None,
                "Ripple (dB)": None,
                "Attenuation (dB)": None,
                "Ripple check": "NO PASSING DESIGN",
                "Attenuation check": "NO PASSING DESIGN",
                "Overall": "FAIL / UNAVAILABLE",
            })
        else:
            passed = result["pass_ok"] and result["stop_ok"]
            rows.append({
                "Design": label,
                "Order": result["order"],
                "Cutoff (Hz)": result["cutoff"],
                "Ripple (dB)": result["ripple"],
                "Attenuation (dB)": result["attenuation"],
                "Ripple check": "PASS" if result["pass_ok"] else "FAIL",
                "Attenuation check": "PASS" if result["stop_ok"] else "FAIL",
                "Overall": "PASS" if passed else "FAIL",
            })

    report = pd.DataFrame(rows)
    st.dataframe(report, use_container_width=True, hide_index=True)

    st.download_button(
        "Download comparison report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="iir_order_comparison.csv",
        mime="text/csv",
    )

    if ai_result is not None:
        with st.expander("AI design numerator coefficients (b)"):
            st.code(np.array2string(ai_result["b"], precision=12))

        with st.expander("AI design denominator coefficients (a)"):
            st.code(np.array2string(ai_result["a"], precision=12))

    if conventional_result is not None:
        with st.expander("Conventional design numerator coefficients (b)"):
            st.code(
                np.array2string(
                    conventional_result["b"], precision=12
                )
            )

        with st.expander("Conventional design denominator coefficients (a)"):
            st.code(
                np.array2string(
                    conventional_result["a"], precision=12
                )
            )

else:
    st.info(
        "Enter the specifications in the sidebar and click "
        "'Predict and compare designs'."
    )
