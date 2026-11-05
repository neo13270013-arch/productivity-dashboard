
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Productivity & Deadline Dashboard",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "productivity_model.joblib"
METADATA_PATH = BASE_DIR / "productivity_metadata.json"
DATA_PATH = BASE_DIR / "cleaned_productivity_data.csv"

st.title("📊 Productivity & Deadline Risk Dashboard")
st.caption(
    "Estimate daily productivity from habits and identify "
    "potential deadline pressure."
)

# Load saved model and metadata
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)
    return model, metadata

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Model files are missing. Place productivity_model.joblib "
        "and productivity_metadata.json next to app.py."
    )
    st.stop()

model, metadata = load_model()

threshold = metadata["low_productivity_threshold"]

# -------------------------------
# SIDEBAR: MODEL INFORMATION
# -------------------------------
st.sidebar.header("Model information")
st.sidebar.write("Algorithm: Random Forest Regression")
st.sidebar.write(
    f"Low-productivity threshold: {threshold:.2f}"
)
st.sidebar.write(
    f"Test MAE: {metadata['mae']:.3f}"
)
st.sidebar.write(
    f"Test RMSE: {metadata['rmse']:.3f}"
)
st.sidebar.write(
    f"Test R²: {metadata['r2']:.3f}"
)

# -------------------------------
# USER INPUTS AND PREDICTION
# -------------------------------
st.header("1. Predict daily productivity")

st.write(
    "Enter the user's daily habits. Fields are generated from "
    "the features used during training."
)

with st.form("productivity_form"):
    user_input = {}

    columns = st.columns(2)

    for i, feature in enumerate(metadata["feature_columns"]):
        spec = metadata["features"][feature]

        with columns[i % 2]:
            if spec["type"] == "numeric":
                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(spec["median"]),
                    help="Enter a value in the units used by your dataset."
                )
            else:
                choices = spec.get("choices", ["Unknown"])
                default = spec.get("default", choices[0])

                if default not in choices:
                    default = choices[0]

                user_input[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=choices,
                    index=choices.index(default)
                )

    st.divider()
    st.subheader("Deadline information")

    days_left = st.number_input(
        "Days remaining until deadline",
        min_value=0,
        max_value=3650,
        value=3
    )

    tasks_remaining = st.number_input(
        "Tasks remaining",
        min_value=0,
        max_value=10000,
        value=3
    )

    submitted = st.form_submit_button("Predict productivity")

if submitted:
    input_df = pd.DataFrame(
        [user_input],
        columns=metadata["feature_columns"]
    )

    try:
        prediction = float(model.predict(input_df)[0])

        # Keep the display within the observed target range.
        # The raw prediction itself is not modified.
        display_prediction = np.clip(
            prediction,
            metadata["target_min"],
            metadata["target_max"]
        )

        st.header("2. Prediction results")

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Predicted productivity",
            f"{display_prediction:.2f}"
        )

        metric2.metric(
            "Low-productivity threshold",
            f"{threshold:.2f}"
        )

        if prediction <= threshold:
            st.warning(
                "Low-productivity warning: the predicted score is "
                "at or below the configured threshold."
            )
        else:
            st.success(
                "The predicted score is above the configured "
                "low-productivity threshold."
            )

        # -------------------------------
        # DEADLINE RISK: RULE-BASED ESTIMATE
        # -------------------------------
        st.header("3. Deadline risk assessment")

        # These are illustrative business rules. They should be
        # validated and adjusted with your project supervisor.
        if tasks_remaining == 0:
            risk = "Low"
            explanation = "No tasks remain unfinished."
        elif days_left == 0:
            risk = "High"
            explanation = "Tasks remain and the deadline is today."
        elif prediction <= threshold and days_left <= 3:
            risk = "High"
            explanation = (
                "Low predicted productivity and a near deadline "
                "coincide with unfinished tasks."
            )
        elif days_left <= 1:
            risk = "High"
            explanation = (
                "The deadline is very close and tasks remain."
            )
        elif prediction <= threshold or days_left <= 3:
            risk = "Medium"
            explanation = (
                "Either the productivity warning or the short "
                "deadline suggests closer monitoring."
            )
        else:
            risk = "Low"
            explanation = (
                "The configured warning conditions are not met. "
                "This does not guarantee on-time completion."
            )

        if risk == "High":
            st.error(f"🔴 HIGH DEADLINE RISK — {explanation}")
        elif risk == "Medium":
            st.warning(f"🟠 MEDIUM DEADLINE RISK — {explanation}")
        else:
            st.success(f"🟢 LOW DEADLINE RISK — {explanation}")

        st.caption(
            "Risk is estimated using simple rules, not a trained "
            "deadline-outcome model. Validate these rules against "
            "real deadline outcomes before operational use."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# -------------------------------
# DATASET OVERVIEW
# -------------------------------
st.header("4. Dataset overview")

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset rows", f"{len(data):,}")
    c2.metric("Dataset columns", len(data.columns))
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")

    st.subheader("Preview of cleaned data")
    st.dataframe(data.head(20), use_container_width=True)

    target = metadata["target_column"]

    if target in data.columns:
        st.subheader("Productivity score distribution")

        scores = pd.to_numeric(data[target], errors="coerce").dropna()

        if not scores.empty:
            st.bar_chart(
                scores.value_counts(bins=10, sort=False)
                .sort_index()
            )

        st.subheader("Average productivity score")

        if not scores.empty:
            st.metric("Dataset average", f"{scores.mean():.2f}")

else:
    st.info(
        "To display dataset statistics, add "
        "cleaned_productivity_data.csv next to app.py."
    )

st.divider()
st.caption(
    "Prototype for productivity planning. Predictions are estimates "
    "and should not be used as definitive judgments of a person."
)
