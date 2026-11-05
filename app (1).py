# =====================================================================
#  Productivity Prediction Dashboard  (CatBoost + Streamlit)
#  Put this file, requirements.txt and everyday_life_dataset.csv
#  in the SAME GitHub folder, then deploy on Streamlit Cloud.
# =====================================================================
import io
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor
import streamlit as st

DATA_FILE = "cleaned_productivity_data.csv"
TARGET = "productivity_rating"

NUM_BASE = ["age", "temperature_c", "sleep_hours", "screen_time_hours",
            "steps", "exercise_minutes", "caffeine_mg", "mood_score", "bedtime_hr"]
CAT_COLS = ["gender", "occupation", "chronotype", "climate_zone",
            "weather", "day_of_week"]
SKEWED = ["steps", "caffeine_mg", "exercise_minutes", "screen_time_hours"]


# ---------------------------------------------------------------------
# 1. DATA + FEATURES
# ---------------------------------------------------------------------
def bedtime_to_hours(value):
    """'02:10' -> 26.17  (hours after midnight of the previous day, so
    00:30 counts as LATER than 23:00 instead of earlier)."""
    if pd.isna(value):
        return np.nan
    h, m = str(value).split(":")[:2]
    x = int(h) + int(m) / 60
    return x + 24 if x < 18 else x


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["bedtime_hr"] = df["bedtime"].map(bedtime_to_hours)
    return df


def make_features(df):
    """Turns raw rows into model inputs. Used for training AND predicting."""
    X = pd.DataFrame(index=df.index)
    for c in NUM_BASE:
        X[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Variable transformation: log1p tames skewed columns ---
    for c in SKEWED:
        X["log_" + c] = np.log1p(X[c].clip(lower=0))

    # --- Engineered features ---
    X["sleep_dev"] = X["sleep_hours"] - 7.5            # distance from ideal sleep
    X["sleep_sq"] = X["sleep_dev"] ** 2                # too little AND too much hurts
    X["caffeine_per_sleep"] = X["caffeine_mg"] / (X["sleep_hours"] + 1)
    X["screen_per_sleep"] = X["screen_time_hours"] / (X["sleep_hours"] + 1)
    X["active_score"] = X["steps"].fillna(0) / 1000 + X["exercise_minutes"].fillna(0) / 10
    X["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    for c in CAT_COLS:
        X[c] = df[c].fillna("Unknown").astype(str)
    return X


# ---------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------
def train_model(df, test_size=0.2, depth=6, learning_rate=0.05):
    X = make_features(df)
    y = df[TARGET].astype(float)

    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42)
    # small validation slice (taken from training data only) for early stopping
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.1, random_state=42)

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=3,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        cat_features=CAT_COLS,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    pred = np.clip(model.predict(X_te), 1, 10)
    importance = pd.Series(model.get_feature_importance(), index=X.columns)

    return {
        "model": model,
        "X_test": X_te,
        "y_test": y_te,
        "pred": pred,
        "r2": r2_score(y_te, pred),
        "mae": mean_absolute_error(y_te, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "importance": importance.sort_values(ascending=False),
        "n_train": len(X_tr_full),
        "n_test": len(X_te),
        "best_iteration": int(model.get_best_iteration() or 0),
    }


# ---------------------------------------------------------------------
# 3. GRAPHS  (each returns a matplotlib figure)
# ---------------------------------------------------------------------
BLUE, ORANGE, GREY = "#2f6fdb", "#e8743b", "#8a8f98"


def plot_actual_vs_pred(y, p, r2, size=(4.6, 3.7)):
    rng = np.random.default_rng(0)
    n = min(2500, len(y))
    idx = rng.choice(len(y), n, replace=False)
    yy = np.asarray(y)[idx] + rng.normal(0, 0.08, n)   # jitter: ratings are whole numbers
    pp = np.asarray(p)[idx]
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(yy, pp, s=7, alpha=0.3, color=BLUE)
    ax.plot([1, 10], [1, 10], "--", color=ORANGE, label="Perfect prediction")
    ax.set_xlabel("Actual productivity")
    ax.set_ylabel("Predicted productivity")
    ax.set_title(f"Actual vs Predicted  (R² = {r2:.3f})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_residuals(y, p, size=(4.6, 3.7)):
    resid = np.asarray(y) - np.asarray(p)
    rng = np.random.default_rng(1)
    n = min(2500, len(resid))
    idx = rng.choice(len(resid), n, replace=False)
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(np.asarray(p)[idx], resid[idx], s=7, alpha=0.3, color=BLUE)
    ax.axhline(0, color=ORANGE, linestyle="--")
    ax.set_xlabel("Predicted productivity")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Residual Plot  (mean {resid.mean():.2f}, std {resid.std():.2f})", fontsize=10)
    fig.tight_layout()
    return fig


def plot_importance(importance, top=12, size=(4.6, 3.9)):
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=size)
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("Importance (%)")
    ax.set_title(f"Feature Importance (top {len(imp)})", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_transformations(df, size=(4.6, 4.6)):
    """Before / after log transformation for two skewed variables."""
    cols = [("steps", "Steps"), ("caffeine_mg", "Caffeine (mg)")]
    fig, axes = plt.subplots(2, 2, figsize=size)
    for j, (c, label) in enumerate(cols):
        raw = df[c].dropna()
        logv = np.log1p(raw.clip(lower=0))
        axes[0, j].hist(raw, bins=40, color=GREY)
        axes[0, j].set_title(f"{label}\nskew = {raw.skew():.2f}", fontsize=8)
        axes[1, j].hist(logv, bins=40, color=BLUE)
        axes[1, j].set_title(f"log({label}) skew = {logv.skew():.2f}", fontsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    fig.suptitle("Variable Transformation (top: raw, bottom: log1p)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_group_means(df, col, size=(5.5, 3.2)):
    g = df.groupby(col)[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=size)
    ax.barh(g.index, g.values, color=BLUE)
    ax.axvline(df[TARGET].mean(), color=ORANGE, linestyle="--", label="Overall average")
    ax.set_xlabel("Average productivity rating")
    ax.set_title(f"Productivity by {col.replace('_', ' ')}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_correlations(df, size=(5.5, 3.6)):
    cols = [c for c in NUM_BASE if c != "age"]
    corr = df[cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=size)
    ax.barh(corr.index, corr.values, color=[ORANGE if v < 0 else BLUE for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with productivity")
    ax.set_title("What moves productivity?", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 4. DOWNLOADABLE REPORTS
# ---------------------------------------------------------------------
def build_pdf_report(res, df, params):
    """Multi-page PDF: summary page + all graphs. Returns bytes."""
    buf = io.BytesIO()
    top3 = ", ".join(res["importance"].head(3).index)
    by_chrono = df.groupby("chronotype")[TARGET].mean().sort_values()
    by_day = df.groupby("day_of_week")[TARGET].mean().sort_values()

    lines = [
        "Productivity Report",
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "",
        "DATASET",
        f"  Records: {len(df):,}    Users: {df['user_id'].nunique():,}",
        f"  Average productivity rating: {df[TARGET].mean():.2f} / 10",
        "",
        "MODEL (CatBoost)",
        f"  R² score: {res['r2']:.3f}",
        f"  MAE: {res['mae']:.3f}    RMSE: {res['rmse']:.3f}",
        f"  Train rows: {res['n_train']:,}    Test rows: {res['n_test']:,}",
        f"  Test size: {params['test_size']:.0%}   Depth: {params['depth']}   "
        f"Learning rate: {params['learning_rate']}",
        f"  Trees used (early stopping): {res['best_iteration']}",
        "",
        "KEY INSIGHTS",
        f"  Top 3 drivers of productivity: {top3}",
        f"  Most productive chronotype: {by_chrono.index[-1]} ({by_chrono.iloc[-1]:.2f})",
        f"  Least productive chronotype: {by_chrono.index[0]} ({by_chrono.iloc[0]:.2f})",
        f"  Best day of the week: {by_day.index[-1]} ({by_day.iloc[-1]:.2f})",
        f"  Weakest day of the week: {by_day.index[0]} ({by_day.iloc[0]:.2f})",
    ]

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, lines[0], fontsize=20, fontweight="bold", va="top")
        y = 0.90
        for line in lines[1:]:
            bold = line.isupper()
            fig.text(0.08, y, line, fontsize=11 if not bold else 12,
                     fontweight="bold" if bold else "normal", va="top",
                     family="monospace" if line.startswith("  ") else None)
            y -= 0.028
        pdf.savefig(fig)
        plt.close(fig)

        figs = [
            plot_actual_vs_pred(res["y_test"], res["pred"], res["r2"], (8, 5.5)),
            plot_residuals(res["y_test"], res["pred"], (8, 5.5)),
            plot_importance(res["importance"], 15, (8, 6)),
            plot_transformations(df, (8, 7)),
            plot_group_means(df, "chronotype", (8, 4)),
            plot_group_means(df, "day_of_week", (8, 4.5)),
            plot_correlations(df, (8, 5)),
        ]
        for f in figs:
            pdf.savefig(f)
            plt.close(f)
    return buf.getvalue()


def build_predictions_csv(res):
    out = res["X_test"].copy()
    out["actual_productivity"] = res["y_test"].values
    out["predicted_productivity"] = np.round(res["pred"], 2)
    out["residual"] = np.round(out["actual_productivity"] - out["predicted_productivity"], 2)
    return out.to_csv(index=False).encode("utf-8")


# =====================================================================
# ===== UI =====  (Streamlit page)
# =====================================================================
st.set_page_config(page_title="Productivity Dashboard", layout="wide")
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:470px;max-width:520px;}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_resource(show_spinner="Training CatBoost model... (first run takes ~1 minute)")
def get_model(test_size, depth, learning_rate):
    return train_model(get_data(), test_size, depth, learning_rate)


@st.cache_data(show_spinner=False)
def get_pdf(test_size, depth, learning_rate):
    params = {"test_size": test_size, "depth": depth, "learning_rate": learning_rate}
    return build_pdf_report(get_model(test_size, depth, learning_rate), get_data(), params)


df = get_data()

# ------------------------- SIDEBAR -----------------------------------
st.sidebar.title("Model Performance")
r2_box = st.sidebar.empty()          # R² is filled in after training

st.sidebar.subheader("Parameters")
test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
depth = st.sidebar.slider("Tree depth", 4, 10, 6)
learning_rate = st.sidebar.select_slider(
    "Learning rate", options=[0.02, 0.03, 0.05, 0.08, 0.10], value=0.05)

res = get_model(test_size, depth, learning_rate)
r2_box.metric("R² score", f"{res['r2']:.3f}",
              delta="target 0.70 reached" if res["r2"] >= 0.70 else f"{res['r2'] - 0.70:+.3f} vs 0.70 target")
c1, c2 = st.sidebar.columns(2)
c1.metric("MAE", f"{res['mae']:.2f}")
c2.metric("RMSE", f"{res['rmse']:.2f}")
st.sidebar.caption(f"Train rows: {res['n_train']:,} | Test rows: {res['n_test']:,} | "
                   f"Trees used: {res['best_iteration']}")

st.sidebar.divider()
st.sidebar.subheader("Model Diagnostics")

for title, make_fig in [
    ("Actual vs Predicted", lambda: plot_actual_vs_pred(res["y_test"], res["pred"], res["r2"])),
    ("Variable Transformation", lambda: plot_transformations(df)),
    ("Residual Plot", lambda: plot_residuals(res["y_test"], res["pred"])),
    ("Feature Importance", lambda: plot_importance(res["importance"])),
]:
    st.sidebar.markdown(f"**{title}**")
    fig = make_fig()
    st.sidebar.pyplot(fig)
    plt.close(fig)

st.sidebar.divider()
st.sidebar.subheader("Download Reports")
st.sidebar.download_button(
    "Download Productivity Report (PDF)",
    data=get_pdf(test_size, depth, learning_rate),
    file_name="productivity_report.pdf",
    mime="application/pdf",
)
st.sidebar.download_button(
    "Download Predictions (CSV)",
    data=build_predictions_csv(res),
    file_name="productivity_predictions.csv",
    mime="text/csv",
)

# ------------------------- MAIN PAGE ---------------------------------
st.title("Productivity Prediction Dashboard")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(df):,}")
k2.metric("Users", f"{df['user_id'].nunique():,}")
k3.metric("Avg productivity", f"{df[TARGET].mean():.2f} / 10")
k4.metric("Model R²", f"{res['r2']:.3f}")

st.subheader("Productivity insights")
a, b = st.columns(2)
with a:
    fig = plot_correlations(df); st.pyplot(fig); plt.close(fig)
    fig = plot_group_means(df, "day_of_week"); st.pyplot(fig); plt.close(fig)
with b:
    fig = plot_group_means(df, "chronotype"); st.pyplot(fig); plt.close(fig)
    fig = plot_group_means(df, "occupation"); st.pyplot(fig); plt.close(fig)

st.subheader("Predict productivity for a day")
med = df[NUM_BASE].median()
p1, p2, p3 = st.columns(3)
with p1:
    sleep = st.slider("Sleep hours", 1.5, 12.0, float(round(med["sleep_hours"], 1)), 0.1)
    bed = st.slider("Bedtime (24 = midnight, 26 = 2 AM)", 20.0, 30.0, float(round(med["bedtime_hr"], 1)), 0.25)
    screen = st.slider("Screen time (hours)", 0.5, 16.5, float(round(med["screen_time_hours"], 1)), 0.1)
    mood = st.slider("Mood score", 1.0, 10.0, float(round(med["mood_score"], 1)), 0.1)
with p2:
    steps = st.number_input("Steps", 0, 42000, int(med["steps"]), 500)
    exercise = st.slider("Exercise minutes", 0, 140, int(med["exercise_minutes"]), 5)
    caffeine = st.slider("Caffeine (mg)", 0, 800, int(med["caffeine_mg"]), 10)
    age = st.slider("Age", 18, 80, int(med["age"]))
with p3:
    chrono = st.selectbox("Chronotype", sorted(df["chronotype"].unique()))
    occ = st.selectbox("Occupation", sorted(df["occupation"].unique()))
    gender = st.selectbox("Gender", sorted(df["gender"].unique()))
    day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                       "Friday", "Saturday", "Sunday"])
    weather = st.selectbox("Weather", sorted(df["weather"].unique()))
    climate = st.selectbox("Climate zone", sorted(df["climate_zone"].unique()))
    temp = st.slider("Temperature (°C)", -15.0, 43.0, float(round(med["temperature_c"], 1)), 0.5)

row = pd.DataFrame([{
    "age": age, "temperature_c": temp, "sleep_hours": sleep, "screen_time_hours": screen,
    "steps": steps, "exercise_minutes": exercise, "caffeine_mg": caffeine,
    "mood_score": mood, "bedtime_hr": bed, "gender": gender, "occupation": occ,
    "chronotype": chrono, "climate_zone": climate, "weather": weather, "day_of_week": day,
}])
pred_one = float(np.clip(res["model"].predict(make_features(row))[0], 1, 10))
st.metric("Predicted productivity rating", f"{pred_one:.1f} / 10")
