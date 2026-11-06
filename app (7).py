# =====================================================================
#  Productivity Prediction Dashboard  (CatBoost + Streamlit)
#  Upload to GitHub (same folder): app.py, requirements.txt,
#  everyday_life_dataset.csv and the files from app_files.zip
#  (model.joblib, model.cbm, metrics.json, test_predictions.csv).
# =====================================================================
import io
import os
import json
import datetime
from math import erf, sqrt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
from catboost import CatBoostRegressor
import streamlit as st

DATA_FILE = "everyday_life_dataset.csv"
TARGET = "productivity_rating"

NUM_BASE = ["age", "temperature_c", "sleep_hours", "screen_time_hours",
            "steps", "exercise_minutes", "caffeine_mg", "mood_score", "bedtime_hr"]
CAT_COLS = ["gender", "occupation", "chronotype", "climate_zone",
            "weather", "day_of_week", "user_id"]
# habits we summarise per user (their "normal")
USER_NUM = ["sleep_hours", "screen_time_hours", "caffeine_mg", "mood_score",
            "steps", "exercise_minutes", "bedtime_hr", "temperature_c"]
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


def compute_user_stats(df):
    """Each user's usual habits (average and spread). Needs a 'user_id' column."""
    g = df.groupby("user_id")[USER_NUM]
    return g.mean().add_prefix("u_").join(g.std().add_prefix("ustd_"))


def make_features(df, user_stats):
    """Turns raw rows into model inputs. Used for training AND predicting.
    A user_id that is not in user_stats (a brand-new user) gets empty habit
    columns, which CatBoost handles natively."""
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

    # --- Personal baseline: how today compares with this user's normal ---
    stats = user_stats.reindex(df["user_id"].values)
    stats.index = df.index
    X = X.join(stats)
    for c in USER_NUM:
        X["dev_" + c] = X[c] - X["u_" + c]

    for c in CAT_COLS:
        X[c] = df[c].fillna("Unknown").astype(str)
    return X


# ---------------------------------------------------------------------
# 2. LOAD THE MODEL TRAINED IN GOOGLE COLAB
#    (files created by productivity_model_colab.ipynb)
# ---------------------------------------------------------------------
REQUIRED_FILES = [DATA_FILE, "metrics.json", "test_predictions.csv"]


def load_model_file():
    """Loads model.joblib; falls back to model.cbm if joblib can't read it."""
    if os.path.exists("model.joblib"):
        try:
            return joblib.load("model.joblib")
        except Exception:
            pass
    model = CatBoostRegressor()
    model.load_model("model.cbm")
    return model


def load_artifacts():
    with open("metrics.json") as f:
        m = json.load(f)
    test = pd.read_csv("test_predictions.csv")
    y_test = test["actual"]
    pred = test["predicted"].to_numpy()
    return {
        "model": load_model_file(),
        "X_test": test.drop(columns=["actual", "predicted"]),
        "y_test": y_test,
        "pred": pred,
        "r2": m["r2"], "mae": m["mae"], "rmse": m["rmse"],
        "importance": pd.Series(m["importance"]).sort_values(ascending=False),
        "n_train": m["n_train"], "n_test": m["n_test"],
        "best_iteration": m["best_iteration"],
        "params": m["params"],
        "cv_r2": m["cv_r2"],
        "train_r2": m["train_r2"],
        "explained_variance": m["explained_variance"],
        "feature_variance": m["feature_variance"],
        "learning_curve": m["learning_curve"],
        "residual_checks": m["residual_checks"],
        "delta_test": m["delta_test"],
        "trained_on": m.get("trained_on", "?"),
        "catboost_version": m.get("catboost_version", "?"),
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


def plot_cv_scores(cv, size=(4.6, 3.4)):
    """R² on each of the cross-validation folds (stable model = similar bars)."""
    cv = list(cv)
    fig, ax = plt.subplots(figsize=size)
    ax.bar([f"Fold {i + 1}" for i in range(len(cv))], cv, color=BLUE)
    ax.axhline(np.mean(cv), color=ORANGE, linestyle="--",
               label=f"Mean {np.mean(cv):.3f} (std {np.std(cv):.3f})")
    ax.set_ylim(max(0, min(cv) - 0.05), min(1, max(cv) + 0.05))
    ax.set_ylabel("R²")
    ax.set_title("Cross-validation R² per fold", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_feature_variance(fv, size=(5.5, 4.4)):
    """Variance of each feature after scaling to 0-1 (near 0 = carries no information)."""
    s = pd.Series(fv).sort_values()
    colors = [ORANGE if v < 0.01 else BLUE for v in s.values]
    fig, ax = plt.subplots(figsize=size)
    ax.barh(s.index, s.values, color=colors)
    ax.axvline(0.01, color="black", linestyle="--", linewidth=0.8, label="Low-variance limit (0.01)")
    ax.set_xlabel("Variance of feature (scaled 0-1)")
    ax.set_title("Feature variance check", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_learning_curve(n_rows, r2s, size=(4.6, 3.4)):
    """R² on the test set when the model is trained on more and more data."""
    fig, ax = plt.subplots(figsize=size)
    ax.plot(n_rows, r2s, marker="o", color=BLUE)
    for x, y_ in zip(n_rows, r2s):
        ax.annotate(f"{y_:.3f}", (x, y_), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7)
    ax.set_xlabel("Training rows used")
    ax.set_ylabel("Test R²")
    ax.set_title("Learning curve (flat end = more data won't help)", fontsize=10)
    ax.set_ylim(min(r2s) - 0.03, max(r2s) + 0.04)
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
def build_pdf_report(res, df):
    """Multi-page PDF: summary page + all graphs. Returns bytes."""
    buf = io.BytesIO()
    params = res["params"]
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
        "VARIANCE CHECK",
        f"  5-fold CV R²: {np.mean(res['cv_r2']):.3f} (std {np.std(res['cv_r2']):.3f})",
        f"  Train R²: {res['train_r2']:.3f}    Test R²: {res['r2']:.3f}    "
        f"Gap: {res['train_r2'] - res['r2']:.3f}",
        f"  Explained variance: {res['explained_variance']:.3f}",
        f"  Low-variance features (< 0.01): "
        f"{', '.join(k for k, v in res['feature_variance'].items() if v < 0.01) or 'none'}",
        "",
        "DATA LIMIT EVIDENCE",
        f"  Learning curve: R² {res['learning_curve']['r2'][0]:.3f} -> {res['learning_curve']['r2'][-1]:.3f} "
        f"(last step {res['learning_curve']['r2'][-1] - res['learning_curve']['r2'][-2]:+.3f})",
        f"  Mean residual: {res['residual_checks']['mean']:.3f}    "
        f"Strongest residual-feature correlation: {res['residual_checks']['max_abs_corr']:.3f}",
        f"  Day-to-day error correlation (lag 1): {res['residual_checks']['lag1_autocorr']:.3f}",
        f"  Estimated noise variance: {res['delta_test']['noise_var']:.2f}    "
        f"Model leftover variance: {res['delta_test']['model_resid_var']:.2f}",
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
            plot_cv_scores(res["cv_r2"], (8, 4.5)),
            plot_feature_variance(res["feature_variance"], (8, 6)),
            plot_learning_curve(res["learning_curve"]["n_rows"], res["learning_curve"]["r2"], (8, 4.5)),
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



# ---------------------------------------------------------------------
# 5. DEADLINE RISK
# ---------------------------------------------------------------------
def deadline_risk(task_hours, days_left, hours_per_day, predicted_rating, avg_rating, rmse):
    """Simple, explainable deadline-risk model.

    * A predicted rating equal to the dataset average = a normal day (efficiency 1.0).
      Better days give efficiency > 1, worse days < 1.
    * Useful hours per day = hours you can work x efficiency.
    * The model's typical error (RMSE) tells us how uncertain each day is.
    * Risk % = chance that total useful hours end up below the hours needed.
    """
    eff = max(predicted_rating / avg_rating, 0.2)
    daily_cap = hours_per_day * eff
    cap_mean = daily_cap * days_left
    daily_sd = hours_per_day * rmse / avg_rating
    cap_sd = max(daily_sd * sqrt(days_left), 1e-6)
    z = (task_hours - cap_mean) / cap_sd
    p_miss = 0.5 * (1 + erf(z / sqrt(2)))

    if p_miss < 0.25:
        label, icon = "Low", "🟢"
    elif p_miss < 0.60:
        label, icon = "Medium", "🟡"
    else:
        label, icon = "High", "🔴"

    days_needed = int(np.ceil(task_hours / daily_cap))
    return {
        "p_miss": float(p_miss), "label": label, "icon": icon,
        "efficiency": float(eff), "daily_capacity": float(daily_cap),
        "capacity_total": float(cap_mean), "daily_sd": float(daily_sd),
        "hours_needed_per_day": float(task_hours / (days_left * eff)),
        "days_needed": days_needed, "extra_days": max(days_needed - days_left, 0),
    }


def plot_deadline(risk, task_hours, days_left, size=(6.5, 3.6)):
    d = np.arange(0, days_left + 1)
    mean = risk["daily_capacity"] * d
    sd = risk["daily_sd"] * np.sqrt(d)
    fig, ax = plt.subplots(figsize=size)
    ax.plot(d, mean, color=BLUE, label="Expected useful hours")
    ax.fill_between(d, np.maximum(mean - sd, 0), mean + sd, color=BLUE, alpha=0.2,
                    label="Likely range (±1 std)")
    ax.axhline(task_hours, color=ORANGE, linestyle="--", label=f"Work needed ({task_hours:g} h)")
    ax.set_xlabel("Days from now")
    ax.set_ylabel("Cumulative useful hours")
    ax.set_title(f"Deadline Risk: {risk['label']} ({risk['p_miss']:.0%} chance of missing)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# =====================================================================
# ===== UI =====  (Streamlit page)
# =====================================================================
st.set_page_config(page_title="Productivity Dashboard", page_icon="📊", layout="wide")
st.markdown(
    "<style>[data-testid='stSidebar']{min-width:470px;max-width:520px;}</style>",
    unsafe_allow_html=True,
)


missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error("These files are missing from your GitHub repo: " + ", ".join(missing) +
             ". Run the Colab notebook, unzip app_files.zip and upload every file next to app.py.")
    st.stop()


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


@st.cache_data(show_spinner=False)
def get_user_stats():
    return compute_user_stats(get_data())


@st.cache_resource(show_spinner="Loading model...")
def get_results():
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_pdf():
    return build_pdf_report(get_results(), get_data())


df = get_data()
res = get_results()
user_stats = get_user_stats()

# ------------------------- SIDEBAR -----------------------------------
st.sidebar.title("📊 Model Performance")
st.sidebar.metric("R² score", f"{res['r2']:.3f}",
                  delta="target 0.70 reached" if res["r2"] >= 0.70 else f"{res['r2'] - 0.70:+.3f} vs 0.70 target")
c1, c2 = st.sidebar.columns(2)
c1.metric("MAE", f"{res['mae']:.2f}")
c2.metric("RMSE", f"{res['rmse']:.2f}")

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
    data=get_pdf(),
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
st.title("📊 Productivity Prediction Dashboard")

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

# ------------------------- VARIANCE CHECK ----------------------------
st.subheader("📉 Variance Check")
cv = res["cv_r2"]
gap = res["train_r2"] - res["r2"]
v1, v2, v3, v4 = st.columns(4)
v1.metric("CV R² (5-fold mean)", f"{np.mean(cv):.3f}")
v2.metric("CV spread (std)", f"± {np.std(cv):.3f}")
v3.metric("Train R²", f"{res['train_r2']:.3f}")
v4.metric("Train - test gap", f"{gap:.3f}")
vc1, vc2 = st.columns(2)
with vc1:
    fig = plot_cv_scores(cv); st.pyplot(fig); plt.close(fig)
with vc2:
    fig = plot_feature_variance(res["feature_variance"]); st.pyplot(fig); plt.close(fig)
low_var = [k for k, v in res["feature_variance"].items() if v < 0.01]
st.caption(
    ("Stable: the score barely changes between different splits of the data. "
     if np.std(cv) < 0.01 else
     "The score changes noticeably between splits, so treat R² as approximate. ")
    + ("The train-test gap is small, so there is no sign of overfitting. "
       if gap < 0.15 else
       "The train-test gap is large, which can mean some overfitting. ")
    + ("Low-variance features: " + ", ".join(low_var) + "." if low_var
       else "No feature is (almost) constant, so every feature can carry information.")
)

# ------------------------- DATA LIMIT EVIDENCE -----------------------
st.subheader("🔬 Data Limit Evidence")
lc, rc, dt = res["learning_curve"], res["residual_checks"], res["delta_test"]
last_gain = lc["r2"][-1] - lc["r2"][-2]
e1, e2 = st.columns(2)
with e1:
    fig = plot_learning_curve(lc["n_rows"], lc["r2"]); st.pyplot(fig); plt.close(fig)
with e2:
    m1, m2 = st.columns(2)
    m1.metric("Mean residual", f"{rc['mean']:.3f}")
    m2.metric("Strongest error-feature link", f"{rc['max_abs_corr']:.3f}", help=rc["max_corr_feature"])
    m3, m4 = st.columns(2)
    m3.metric("Day-to-day error link (lag 1)", f"{rc['lag1_autocorr']:.3f}")
    m4.metric("Last learning-curve step", f"{last_gain:+.3f}")
    m5, m6 = st.columns(2)
    m5.metric("Estimated noise variance", f"{dt['noise_var']:.2f}")
    m6.metric("Model leftover variance", f"{dt['model_resid_var']:.2f}")
st.caption(
    ("The learning curve has flattened, so more data of this kind is unlikely to lift R² much. "
     if last_gain < 0.01 else
     "The learning curve is still rising, so more data could help. ")
    + (f"The errors have no strong link with any feature (largest {rc['max_abs_corr']:.2f}), "
       "so the model is not missing an obvious pattern in the data it has. "
       if rc["max_abs_corr"] < 0.05 else
       f"The errors still correlate with {rc['max_corr_feature']}, so that feature may deserve more work. ")
    + "The noise estimate (nearest-neighbour test on the same user's similar days) is an upper bound: "
      "when the model's leftover variance is close to it, little is left to learn. "
      "This is strong evidence of a data limit, not a mathematical proof."
)

# ------------------------- PREDICTOR ---------------------------------
st.subheader("Predict productivity for a day")
NEW_USER = "New user (no history)"
who = st.selectbox("Whose day?", [NEW_USER] + sorted(df["user_id"].unique()))
mine = df if who == NEW_USER else df[df["user_id"] == who]
med = mine[NUM_BASE].median().fillna(df[NUM_BASE].median())


def clamp(v, lo, hi):
    return float(min(max(v, lo), hi))


def pick(label, col):
    options = sorted(df[col].unique())
    default = mine[col].mode().iloc[0]
    return st.selectbox(label, options, index=options.index(default), key=f"{col}_{who}")


p1, p2, p3 = st.columns(3)
with p1:
    sleep = st.slider("Sleep hours", 1.5, 12.0, clamp(round(med["sleep_hours"], 1), 1.5, 12.0), 0.1, key=f"sleep_{who}")
    bed = st.slider("Bedtime (24 = midnight, 26 = 2 AM)", 20.0, 30.0, clamp(round(med["bedtime_hr"], 1), 20.0, 30.0), 0.25, key=f"bed_{who}")
    screen = st.slider("Screen time (hours)", 0.5, 16.5, clamp(round(med["screen_time_hours"], 1), 0.5, 16.5), 0.1, key=f"screen_{who}")
    mood = st.slider("Mood score", 1.0, 10.0, clamp(round(med["mood_score"], 1), 1.0, 10.0), 0.1, key=f"mood_{who}")
with p2:
    steps = st.number_input("Steps", 0, 42000, int(clamp(med["steps"], 0, 42000)), 500, key=f"steps_{who}")
    exercise = st.slider("Exercise minutes", 0, 140, int(clamp(med["exercise_minutes"], 0, 140)), 5, key=f"ex_{who}")
    caffeine = st.slider("Caffeine (mg)", 0, 800, int(clamp(med["caffeine_mg"], 0, 800)), 10, key=f"caf_{who}")
    age = st.slider("Age", 18, 80, int(clamp(med["age"], 18, 80)), key=f"age_{who}")
with p3:
    chrono = pick("Chronotype", "chronotype")
    occ = pick("Occupation", "occupation")
    gender = pick("Gender", "gender")
    day = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                       "Friday", "Saturday", "Sunday"])
    weather = pick("Weather", "weather")
    climate = pick("Climate zone", "climate_zone")
    temp = st.slider("Temperature (°C)", -15.0, 43.0, clamp(round(med["temperature_c"], 1), -15.0, 43.0), 0.5, key=f"temp_{who}")

row = pd.DataFrame([{
    "user_id": "NEW" if who == NEW_USER else who,
    "age": age, "temperature_c": temp, "sleep_hours": sleep, "screen_time_hours": screen,
    "steps": steps, "exercise_minutes": exercise, "caffeine_mg": caffeine,
    "mood_score": mood, "bedtime_hr": bed, "gender": gender, "occupation": occ,
    "chronotype": chrono, "climate_zone": climate, "weather": weather, "day_of_week": day,
}])
pred_one = float(np.clip(res["model"].predict(make_features(row, user_stats))[0], 1, 10))
st.metric("Predicted productivity rating", f"{pred_one:.1f} / 10")


# ------------------------- DEADLINE RISK -----------------------------
st.subheader("⏰ Deadline Risk")
st.caption("Uses the predicted productivity from the panel above "
           f"({pred_one:.1f} / 10) to estimate whether you will finish on time.")
d1, d2, d3 = st.columns(3)
task_hours = d1.number_input("Hours of work needed", 1.0, 500.0, 20.0, 1.0)
days_left = int(d2.number_input("Days until deadline", 1, 60, 5, 1))
hours_pd = d3.slider("Hours you can work per day", 1.0, 12.0, 6.0, 0.5)

risk = deadline_risk(task_hours, days_left, hours_pd, pred_one,
                     float(df[TARGET].mean()), res["rmse"])

r1, r2c, r3 = st.columns(3)
r1.metric("Risk of missing deadline", f"{risk['icon']} {risk['label']}", f"{risk['p_miss']:.0%}", delta_color="off")
r2c.metric("Useful hours per day", f"{risk['daily_capacity']:.1f} h",
           f"{risk['efficiency']:.0%} of a normal day", delta_color="off")
r3.metric("Hours/day needed", f"{risk['hours_needed_per_day']:.1f} h")
st.progress(min(max(risk["p_miss"], 0.0), 1.0))

if risk["label"] == "High":
    st.error(f"At this productivity level you would need about {risk['days_needed']} days "
             f"({risk['extra_days']} more than you have), or {risk['hours_needed_per_day']:.1f} "
             "working hours per day.")
elif risk["label"] == "Medium":
    st.warning("It is tight. A low-productivity day (poor sleep, lots of caffeine or screen time) "
               "could push you past the deadline. Try to build in a buffer day.")
else:
    st.success("You are on track, even allowing for some bad days.")

fig = plot_deadline(risk, task_hours, days_left)
st.pyplot(fig)
plt.close(fig)
