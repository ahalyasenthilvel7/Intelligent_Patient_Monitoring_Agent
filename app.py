
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

from streamlit_autorefresh import st_autorefresh

from simulator import generate_reading
from train_model import train_models
from agent import build_agent


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Intelligent Patient Monitoring Agent",
    page_icon="🏥",
    layout="wide"
)


# TRAIN OR LOAD MODEL

@st.cache_resource
def load_system():

    if (
        os.path.exists("models/patient_model.pkl")
        and os.path.exists("models/label_encoder.pkl")
    ):

        model = joblib.load(
            "models/patient_model.pkl"
        )

        encoder = joblib.load(
            "models/label_encoder.pkl"
        )

        results = {}

        selected = "Previously trained model"

    else:

        trained = train_models()

        model = trained["model"]
        encoder = trained["encoder"]
        results = trained["results"]
        selected = trained["selected"]

    graph = build_agent()

    return model, encoder, graph, results, selected


model, encoder, graph, results, selected = load_system()


# SESSION STATE

if "history" not in st.session_state:
    st.session_state.history = []

if "activities" not in st.session_state:
    st.session_state.activities = []

if "latest" not in st.session_state:
    st.session_state.latest = None


# HEADER

st.title("🏥 Intelligent Patient Monitoring Agent")

st.caption(
    "Agentic AI + Machine Learning + Real-Time Monitoring"
)

st.warning(
    "Educational demonstration using synthetic data. "
    "Not intended for diagnosis, clinical decisions, "
    "or actual patient monitoring."
)


# SIDEBAR

st.sidebar.title("Monitoring Settings")

patient_id = st.sidebar.text_input(
    "Patient ID",
    "PATIENT_001"
)

mode = st.sidebar.selectbox(
    "Simulation Mode",
    ["Auto", "Normal", "Unusual"]
)

refresh = st.sidebar.checkbox(
    "Enable automatic updates",
    value=True
)

if refresh:
    st_autorefresh(
        interval=5000,
        limit=None,
        key="patient_refresh"
    )


# GENERATE READINGS

reading = generate_reading(mode)

history_before = st.session_state.history.copy()

state = graph.invoke({

    "readings": reading,

    "history": history_before,

    "model": model,

    "encoder": encoder

})


# SAVE RESULTS

result = {

    **reading,

    "prediction": state["prediction"],

    "confidence": state["confidence"],

    "priority": state["priority"]

}

st.session_state.latest = result

st.session_state.history.append(result)

st.session_state.activities.extend(
    state.get("activity", [])
)

# Keep recent history
st.session_state.history = (
    st.session_state.history[-100:]
)

st.session_state.activities = (
    st.session_state.activities[-50:]
)


# PATIENT INFORMATION

st.subheader("Patient Information")

st.write("Patient ID:", patient_id)

st.write("Monitoring Mode:", mode)


# VITAL SIGNS

st.subheader("Live Patient Readings")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Heart Rate",
    f"{reading['heart_rate']} BPM"
)

c2.metric(
    "SpO2",
    f"{reading['spo2']}%"
)

c3.metric(
    "Temperature",
    f"{reading['temperature']} °C"
)

c4, c5, c6 = st.columns(3)

c4.metric(
    "Systolic BP",
    reading["systolic_bp"]
)

c5.metric(
    "Diastolic BP",
    reading["diastolic_bp"]
)

c6.metric(
    "Respiratory Rate",
    reading["respiratory_rate"]
)


# ML PREDICTION

st.divider()

st.subheader("Machine Learning Analysis")

st.write("Selected model:", selected)

st.write(
    "Model prediction:",
    state["prediction"]
)

st.write(
    "Model confidence:",
    f"{state['confidence'] * 100:.2f}%"
)

st.caption(
    "Confidence represents the model's predicted "
    "class probability, not clinical certainty."
)


# AGENT DECISION

st.subheader("Agentic AI Decision")

st.write(
    "Priority:",
    state["priority"]
)

st.info(state["message"])


# AGENT ACTIVITY

st.subheader("Agent Activity")

for activity in state.get("activity", []):

    st.write("✓", activity)


# HISTORY CHART

st.divider()

st.subheader("Patient History")

history_df = pd.DataFrame(
    st.session_state.history
)

if not history_df.empty:

    chart = px.line(
        history_df,
        x="timestamp",
        y="heart_rate",
        title="Heart Rate Monitoring",
        markers=True
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    chart2 = px.line(
        history_df,
        x="timestamp",
        y="spo2",
        title="SpO2 Monitoring",
        markers=True
    )

    st.plotly_chart(
        chart2,
        use_container_width=True
    )

    st.subheader("Recent Patient Readings")

    st.dataframe(
        history_df.tail(10),
        use_container_width=True
    )


# ALERT HISTORY

st.divider()

st.subheader("Alert History")

alerts = history_df[
    history_df["prediction"] != "Normal"
]

if not alerts.empty:

    st.dataframe(
        alerts[
            [
                "timestamp",
                "prediction",
                "priority",
                "confidence"
            ]
        ].tail(10),
        use_container_width=True
    )

else:

    st.success("No unusual readings flagged in this session.")


# MODEL COMPARISON

if results:

    st.divider()

    st.subheader("ML Model Comparison")

    comparison = pd.DataFrame(results).T

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )


# FOOTER

st.divider()

st.caption(
    "Intelligent Patient Monitoring Agent | "
    "Educational AI Prototype"
)
