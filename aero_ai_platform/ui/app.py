# aero_ai_platform/ui/dashboard_streamlit.py

from pathlib import Path
import sys

import streamlit as st
import pandas as pd

# Add parent directory to path so imports work when running directly
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from aero_ai_platform.config import (
    FlightConfig,
    EngineConfig,
    SHMConfig,
)

# Data generation
from aero_ai_platform.data_generation.flight import save_flight_dataset
from aero_ai_platform.data_generation.engine import save_engine_dataset
from aero_ai_platform.data_generation.shm import save_shm_dataset

# Feature builders & raw loaders
from aero_ai_platform.features.flight_features import (
    build_flight_features,
    load_flight_raw,
)
from aero_ai_platform.features.engine_features import (
    build_engine_features,
    load_engine_raw,
)
from aero_ai_platform.features.shm_features import (
    build_shm_features,
    load_shm_raw,
)

# Models
from aero_ai_platform.models.flight import load_flight_model, train_flight_model
from aero_ai_platform.models.engine import load_engine_model, train_engine_model
from aero_ai_platform.models.shm import load_shm_model, train_shm_model


# ----------------- Streamlit setup ----------------- #

st.set_page_config(
    page_title="Aero AI Health & Safety Dashboard (MVP)",
    layout="wide",
)


# ----------------- Caching helpers ----------------- #

@st.cache_data
def get_flight_raw() -> pd.DataFrame:
    try:
        return load_flight_raw()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_data
def get_flight_features() -> pd.DataFrame:
    try:
        return build_flight_features(get_flight_raw())
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_data
def get_engine_raw() -> pd.DataFrame:
    try:
        return load_engine_raw()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_data
def get_engine_features() -> pd.DataFrame:
    try:
        return build_engine_features(get_engine_raw())
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_data
def get_shm_raw() -> pd.DataFrame:
    try:
        return load_shm_raw()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_data
def get_shm_features() -> pd.DataFrame:
    try:
        return build_shm_features(get_shm_raw())
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource
def get_flight_model():
    try:
        return load_flight_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource
def get_engine_model():
    try:
        return load_engine_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


@st.cache_resource
def get_shm_model():
    try:
        return load_shm_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()


# ----------------- Interpretation helpers ----------------- #

def _risk_bucket_from_probability(p: float) -> str:
    """
    Generic mapping from probability to a human-friendly risk level.
    """
    if p < 0.25:
        return f"Low risk ({p:.2f})"
    elif p < 0.6:
        return f"Medium risk ({p:.2f})"
    else:
        return f"High risk ({p:.2f})"


def _highlight_deviations(row: pd.Series, all_rows: pd.DataFrame, feature_map: dict) -> list:
    """
    Compare a single row to fleet/global statistics and generate explanations
    for features that are significantly higher/lower than usual.
    feature_map: {column_name: human_readable_label}
    """
    msgs = []
    stats = all_rows[feature_map.keys()].describe()

    for col, label in feature_map.items():
        if col not in row.index:
            continue

        mean = stats.loc["mean", col]
        std = stats.loc["std", col]
        if std == 0 or pd.isna(std):
            continue

        z = (row[col] - mean) / std

        if z > 1.0:
            msgs.append(f"- **{label}** is higher than typical (z≈{z:.1f}).")
        elif z < -1.0:
            msgs.append(f"- **{label}** is lower than typical (z≈{z:.1f}).")

    return msgs


def _extract_feature_labels_from_msgs(explanation_points: list[str]) -> list[str]:
    """
    From messages like '- **Altitude variability** is higher than typical'
    extract 'Altitude variability' etc., without duplicates.
    """
    labels = []
    for msg in explanation_points:
        start = msg.find("**")
        if start == -1:
            continue
        end = msg.find("**", start + 2)
        if end == -1:
            continue
        label = msg[start + 2:end].strip()
        if label and label not in labels:
            labels.append(label)
    return labels


# ----------------- Model prediction helpers ----------------- #

def _predict_single_flight(flight_id: int):
    model, feature_names = get_flight_model()
    df_feat = get_flight_features()
    row = df_feat[df_feat["flight_id"] == flight_id]
    if row.empty:
        return None, None, None

    X = row[feature_names]
    proba = model.predict_proba(X)[0]  # [p_normal, p_anomaly]
    pred_label = "Anomalous flight" if proba[1] >= 0.5 else "Normal flight"
    risk_text = _risk_bucket_from_probability(proba[1])

    # Interpret using key features
    row_series = row.iloc[0]
    explanation_points = []

    key_features = {
        "approach_alt_std": "Altitude variability on approach",
        "approach_ias_std": "Speed variability on approach",
        "alt_std": "Overall altitude variability",
        "ias_std": "Overall speed variability",
        "pitch_std": "Pitch attitude variability",
        "roll_std": "Bank angle variability",
    }

    deviation_msgs = _highlight_deviations(row_series, df_feat, key_features)
    if deviation_msgs:
        explanation_points.extend(deviation_msgs)

    if not explanation_points:
        explanation_points.append("- Flight parameters are within typical ranges for the fleet.")

    return pred_label, risk_text, explanation_points


def _predict_engine_rul_for_engine(engine_id: int):
    model, feature_names = get_engine_model()
    df_feat = get_engine_features()
    df_eng = df_feat[df_feat["engine_id"] == engine_id].copy()
    if df_eng.empty:
        return None, None, None, None

    X = df_eng[feature_names]
    y_true = df_eng["rul_cycles"]
    y_pred = model.predict(X)

    # latest cycle
    latest = df_eng.sort_values("cycle").iloc[-1]
    X_last = latest[feature_names].to_frame().T
    y_last_pred = model.predict(X_last)[0]

    return df_eng["cycle"].values, y_true.values, y_pred, (latest, y_last_pred)


def _engine_health_bucket(rul_pred: float, all_rul: pd.Series) -> str:
    """
    Turn RUL into a simple health label: Healthy / Monitor / Warning / Critical.
    """
    q25 = all_rul.quantile(0.25)
    q50 = all_rul.quantile(0.5)
    q75 = all_rul.quantile(0.75)

    if rul_pred >= q75:
        return f"Healthy (high remaining life ≈ {rul_pred:.0f} cycles)"
    elif rul_pred >= q50:
        return f"Monitor (moderate remaining life ≈ {rul_pred:.0f} cycles)"
    elif rul_pred >= q25:
        return f"Warning (low remaining life ≈ {rul_pred:.0f} cycles)"
    else:
        return f"Critical (very low remaining life ≈ {rul_pred:.0f} cycles)"


def _explain_engine_latest(latest_row: pd.Series, df_all: pd.DataFrame) -> list:
    msgs = []

    key_features = {
        "egt_c": "Exhaust gas temperature",
        "fuel_flow_kgph": "Fuel flow",
        "n1_pct": "N1 speed",
    }

    msgs.extend(_highlight_deviations(latest_row, df_all, key_features))

    if not msgs:
        msgs.append("- Engine temperature, core speed and fuel flow look typical compared to other engines.")

    return msgs


def _predict_shm_sensor(sensor_id: int):
    model, feature_names = get_shm_model()
    df_feat = get_shm_features()
    row = df_feat[df_feat["sensor_id"] == sensor_id]
    if row.empty:
        return None, None, None

    X = row[feature_names]
    proba = model.predict_proba(X)[0]  # [p_healthy, p_damaged]
    pred_label = "Damaged structure" if proba[1] >= 0.5 else "Healthy structure"
    risk_text = _risk_bucket_from_probability(proba[1])

    row_series = row.iloc[0]
    explanation_points = []

    key_features = {
        "rms": "Overall vibration level (RMS)",
        "max_abs": "Peak vibration amplitude",
        "hf_ratio": "High-frequency vibration content",
        "energy": "Total vibration energy",
    }

    deviation_msgs = _highlight_deviations(row_series, df_feat, key_features)
    if deviation_msgs:
        explanation_points.extend(deviation_msgs)

    if not explanation_points:
        explanation_points.append("- Vibration pattern is within normal range compared to other sensors.")

    return pred_label, risk_text, explanation_points


# ----------------- Plain-language summary helpers ----------------- #

def _plain_text_summary_flight(
    flight_id: int,
    pred_label: str,
    risk_text: str,
    explanation_points: list[str],
    true_label: str,
) -> str:
    risk_level = risk_text.split()[0] if risk_text else "Unknown"
    key_factors = _extract_feature_labels_from_msgs(explanation_points)
    if key_factors:
        factor_text = ", ".join(key_factors[:3])
    else:
        factor_text = "overall flight parameters"

    return (
        f"Flight {flight_id} is assessed as **{pred_label.lower()}** with **{risk_level.lower()}** risk. "
        f"The model is mainly reacting to **{factor_text}** compared to other flights in the fleet. "
        f"(Synthetic ground truth label for this flight is **{true_label}**.)"
    )


def _plain_text_summary_engine(
    engine_id: int,
    latest_cycle: int,
    health_text: str,
    latest_row: pd.Series,
    explanation_points: list[str],
) -> str:
    health_level = health_text.split()[0] if health_text else "Unknown"
    key_factors = _extract_feature_labels_from_msgs(explanation_points)
    if key_factors:
        factor_text = ", ".join(key_factors[:3])
    else:
        factor_text = "temperature, fuel flow and core speed"

    return (
        f"Engine {engine_id} at cycle {latest_cycle} is in **{health_level.lower()}** condition "
        f"based on its remaining useful life. The assessment is mainly influenced by **{factor_text}**, "
        f"compared to other engines in the fleet."
    )


def _plain_text_summary_shm(
    sensor_id: int,
    pred_label: str,
    risk_text: str,
    explanation_points: list[str],
    true_label: str,
) -> str:
    risk_level = risk_text.split()[0] if risk_text else "Unknown"
    key_factors = _extract_feature_labels_from_msgs(explanation_points)
    if key_factors:
        factor_text = ", ".join(key_factors[:3])
    else:
        factor_text = "overall vibration behaviour"

    return (
        f"Sensor {sensor_id} indicates **{pred_label.lower()}** with **{risk_level.lower()}** risk. "
        f"The model focuses mainly on **{factor_text}** when comparing this location with other sensors. "
        f"(Synthetic ground truth label for this sensor is **{true_label}**.)"
    )


# ----------------- Layout ----------------- #

st.title("✈️ Aero AI Health & Safety Dashboard (Synthetic MVP)")
st.caption(
    "Offline, synthetic-data MVP for Flight Operations, Engine Health, and Structural Health Monitoring.\n"
    "All model outputs are translated into simple, human-understandable risk descriptions and short reports."
)

# ----------------------------------------------------
# Top-level controls: data generation & model training
# ----------------------------------------------------
with st.expander("Data & model lifecycle controls", expanded=True):
    st.markdown(
        "Use these actions to (re)generate synthetic CSV datasets and (re)train the models directly from the app."
    )

    col_setup1, col_setup2 = st.columns(2)

    with col_setup1:
        st.subheader("Step 1 – Generate synthetic datasets")
        st.write(
            "This will create or overwrite the synthetic flight, engine and SHM CSV files "
            f"in `{FlightConfig.OUTPUT_PATH.parent}`."
        )

        if st.button("Generate synthetic CSV files", key="btn_generate_data"):
            with st.spinner("Generating synthetic datasets..."):
                save_flight_dataset()
                save_engine_dataset()
                save_shm_dataset()

            # Clear cached data & features so fresh CSVs are used
            get_flight_raw.clear()
            get_flight_features.clear()
            get_engine_raw.clear()
            get_engine_features.clear()
            get_shm_raw.clear()
            get_shm_features.clear()

            st.success("Synthetic datasets generated. Tabs below now use the new CSV files.")

    with col_setup2:
        st.subheader("Step 2 – Train models")
        st.write(
            "Train or retrain the three models using the current synthetic datasets and save them "
            f"into `{EngineConfig.OUTPUT_PATH.parent.parent / 'models_artifacts'}`."
        )

        if st.button("Train all models", key="btn_train_models"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Training Flight anomaly model...")
                progress_bar.progress(33)
                train_flight_model()
                
                status_text.text("Training Engine RUL model...")
                progress_bar.progress(66)
                train_engine_model()
                
                status_text.text("Training SHM damage model...")
                progress_bar.progress(100)
                train_shm_model()
                
                # Clear cached models so future calls reload updated artifacts
                get_flight_model.clear()
                get_engine_model.clear()
                get_shm_model.clear()
                
                progress_bar.empty()
                status_text.empty()
                st.success("All models trained and saved. Tabs below now use the latest models.")
            except FileNotFoundError as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Training failed: {str(e)}. Please generate datasets first.")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Training failed with error: {str(e)}")

st.markdown("---")

tab_flight, tab_engine, tab_shm = st.tabs(["🛫 Flight Operations", "🛠 Engine Health", "🧱 Structural Health"])


# ====================================================
#                 FLIGHT TAB
# ====================================================
with tab_flight:
    st.header("Flight Operations Monitoring (FOQA / FDM)")

    df_raw = get_flight_raw()
    df_feat = get_flight_features()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Flight")
        flight_ids = sorted(df_raw["flight_id"].unique().tolist())
        selected_flight = st.selectbox("Flight ID", flight_ids)

        df_flight = df_raw[df_raw["flight_id"] == selected_flight].sort_values("time_idx")

        st.markdown("**Flight profile (Altitude & IAS vs Time)**")
        plot_df = df_flight[["time_idx", "altitude_m", "ias_kt"]].set_index("time_idx")
        st.line_chart(plot_df)

        st.markdown("**Attitude & Engine N1 vs Time**")
        plot_df2 = df_flight[["time_idx", "pitch_deg", "roll_deg", "engine_n1_pct"]].set_index("time_idx")
        st.line_chart(plot_df2)

    with col2:
        st.subheader("Model Inference – Human Explanation")

        pred_label, risk_text, explanation_points = _predict_single_flight(selected_flight)

        true_label = df_feat.loc[df_feat["flight_id"] == selected_flight, "anomaly_label"].iloc[0]

        st.metric(
            label="True label (synthetic ground truth)",
            value=true_label,
        )

        if pred_label is not None:
            st.metric(
                label="AI assessment",
                value=pred_label,
                delta=risk_text,
            )

            st.markdown("**Why the model thinks this:**")
            for msg in explanation_points:
                st.markdown(msg)

            if st.button("Explain in plain words", key="flight_explain"):
                summary = _plain_text_summary_flight(
                    flight_id=selected_flight,
                    pred_label=pred_label,
                    risk_text=risk_text,
                    explanation_points=explanation_points,
                    true_label=true_label,
                )
                st.info(summary)
        else:
            st.error("No feature row found for this flight.")

        st.markdown("---")
        st.markdown("**Feature snapshot for this flight (for analysts)**")
        st.dataframe(df_feat[df_feat["flight_id"] == selected_flight])


# ====================================================
#                 ENGINE TAB
# ====================================================
with tab_engine:
    st.header("Engine Health & Remaining Useful Life (RUL)")

    df_raw_eng = get_engine_raw()
    df_feat_eng = get_engine_features()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Engine")
        engine_ids = sorted(df_raw_eng["engine_id"].unique().tolist())
        selected_engine = st.selectbox("Engine ID", engine_ids)

        cycles, y_true, y_pred, latest_info = _predict_engine_rul_for_engine(selected_engine)

        if cycles is not None:
            st.markdown("**RUL prediction vs true (synthetic)**")
            df_plot = pd.DataFrame(
                {
                    "cycle": cycles,
                    "true_rul": y_true,
                    "predicted_rul": y_pred,
                }
            ).set_index("cycle")
            st.line_chart(df_plot)

            st.markdown("**Engine parameters vs cycles**")
            df_eng_series = df_feat_eng[df_feat_eng["engine_id"] == selected_engine].set_index("cycle")
            st.line_chart(df_eng_series[["egt_c", "n1_pct", "fuel_flow_kgph"]])
        else:
            st.error("No data found for this engine.")

    with col2:
        st.subheader("Current Engine Health – Human Explanation")

        if latest_info is not None:
            latest_row, y_last_pred = latest_info

            st.metric("Latest cycle", int(latest_row["cycle"]))
            st.metric("EGT (°C)", f"{latest_row['egt_c']:.1f}")
            st.metric("N1 (%)", f"{latest_row['n1_pct']:.1f}")
            st.metric("Fuel flow (kg/h)", f"{latest_row['fuel_flow_kgph']:.1f}")
            st.metric("True RUL (synthetic)", f"{latest_row['rul_cycles']:.1f}")

            health_text = _engine_health_bucket(
                rul_pred=y_last_pred,
                all_rul=df_feat_eng["rul_cycles"],
            )
            st.metric("Predicted RUL (model)", f"{y_last_pred:.1f}", delta=health_text)

            expl = _explain_engine_latest(latest_row, df_feat_eng)

            st.markdown("**Why the model thinks this:**")
            for msg in expl:
                st.markdown(msg)

            if st.button("Explain in plain words", key="engine_explain"):
                summary = _plain_text_summary_engine(
                    engine_id=selected_engine,
                    latest_cycle=int(latest_row["cycle"]),
                    health_text=health_text,
                    latest_row=latest_row,
                    explanation_points=expl,
                )
                st.info(summary)
        else:
            st.info("Select an engine with available cycles to see health explanation.")

        st.markdown("---")
        st.markdown("**Raw feature row (latest cycle)**")
        if latest_info is not None:
            latest_row, _ = latest_info
            st.dataframe(latest_row.to_frame().T)


# ====================================================
#                 SHM TAB
# ====================================================
with tab_shm:
    st.header("Structural Health Monitoring (Vibration-based)")

    df_raw_shm = get_shm_raw()
    df_feat_shm = get_shm_features()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Select Sensor")
        sensor_ids = sorted(df_raw_shm["sensor_id"].unique().tolist())
        selected_sensor = st.selectbox("Sensor ID", sensor_ids)

        df_sensor = df_raw_shm[df_raw_shm["sensor_id"] == selected_sensor].sort_values("time_idx")

        st.markdown("**Vibration signal (first 1000 samples)**")
        max_samples = min(1000, len(df_sensor))
        small_seg = df_sensor.iloc[:max_samples][["time_idx", "accel_g"]].set_index("time_idx")
        st.line_chart(small_seg)

    with col2:
        st.subheader("Model Inference – Human Explanation")

        pred_label, risk_text, explanation_points = _predict_shm_sensor(selected_sensor)
        true_label = df_feat_shm.loc[
            df_feat_shm["sensor_id"] == selected_sensor, "damage_label"
        ].iloc[0]

        st.metric("True label (synthetic)", true_label)

        if pred_label is not None:
            st.metric(
                "AI assessment",
                pred_label,
                delta=risk_text,
            )

            st.markdown("**Why the model thinks this:**")
            for msg in explanation_points:
                st.markdown(msg)

            if st.button("Explain in plain words", key="shm_explain"):
                summary = _plain_text_summary_shm(
                    sensor_id=selected_sensor,
                    pred_label=pred_label,
                    risk_text=risk_text,
                    explanation_points=explanation_points,
                    true_label=true_label,
                )
                st.info(summary)
        else:
            st.error("No feature row for this sensor.")

        st.markdown("---")
        st.markdown("**Feature snapshot for this sensor**")
        st.dataframe(df_feat_shm[df_feat_shm["sensor_id"] == selected_sensor])
