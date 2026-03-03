# -*- coding: utf-8 -*-
"""
Anomaly Detection + SPC Dashboard (Demo + Upload + Column Mapping)
Created on Tue Mar  3 2026
@author: chad
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

import plotly.graph_objects as go

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Anomaly + SPC Dashboard", layout="wide")
st.title("Manufacturing Anomaly Detection + SPC Dashboard")

# ----------------------------
# Demo + helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def make_demo_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    ts0 = pd.Timestamp("2026-01-01 06:00:00")

    df = pd.DataFrame({
        "Timestamp": ts0 + pd.to_timedelta(np.arange(n) * 15, unit="min"),
        "BatchID": [f"B{str(i).zfill(5)}" for i in range(n)],
        "MaterialLot": rng.choice(["MLOT1", "MLOT2", "MLOT3", "MLOT4"], size=n),
        "Machine": rng.choice(["M1", "M2", "M3"], size=n, p=[0.45, 0.35, 0.20]),
        "Operator": rng.choice(["O1", "O2", "O3", "O4"], size=n),
        "Shift": np.where((np.arange(n) % 96) < 48, "Day", "Night"),
        "Temperature": np.round(rng.normal(200, 6.5, size=n), 2),
        "Pressure": np.round(rng.normal(5.0, 0.55, size=n), 2),
        "Speed": np.round(rng.normal(100, 7, size=n), 2),
    })

    # Inject drift (FIXED: avoid broadcasting mismatch)
    start = int(n * 0.65)
    mask = df.index >= start
    df.loc[mask, "Temperature"] += np.linspace(0, 8, mask.sum())

    # Inject spikes / anomalies
    spike_idx = rng.choice(np.arange(n), size=max(6, n // 60), replace=False)
    df.loc[spike_idx, "Pressure"] += rng.normal(1.8, 0.3, size=len(spike_idx))
    df.loc[spike_idx, "Machine"] = "M3"
    df.loc[spike_idx, "Shift"] = "Night"

    return df

@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(s))

def apply_mapping(df_raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename = {src: dst for dst, src in mapping.items() if src and src != "—"}
    return df_raw.rename(columns=rename)

def minmax_0_100(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(x)
    return 100.0 * (x - lo) / (hi - lo)

# ----------------------------
# Nelson rules (basic)
# ----------------------------
def nelson_rule_2_same_side(x: np.ndarray, mean: float, k: int = 9) -> np.ndarray:
    side = np.where(x >= mean, 1, -1)
    out = np.zeros(len(x), dtype=bool)
    run = 1
    for i in range(1, len(x)):
        if side[i] == side[i - 1]:
            run += 1
        else:
            run = 1
        if run >= k:
            out[i] = True
    return out

def nelson_rule_3_trend(x: np.ndarray, k: int = 6) -> np.ndarray:
    out = np.zeros(len(x), dtype=bool)
    if len(x) < k:
        return out
    for i in range(k - 1, len(x)):
        window = x[i - k + 1 : i + 1]
        diffs = np.diff(window)
        if np.all(diffs > 0) or np.all(diffs < 0):
            out[i] = True
    return out

def spc_flags_for_series(x: pd.Series, use_robust: bool = False):
    arr = x.to_numpy(dtype=float)
    valid = np.isfinite(arr)

    if not valid.any():
        m = np.nan
        s = np.nan
    else:
        if use_robust:
            med = np.nanmedian(arr)
            mad = np.nanmedian(np.abs(arr - med))
            s = 1.4826 * mad if mad > 0 else np.nanstd(arr, ddof=1)
            m = med
        else:
            m = np.nanmean(arr)
            s = np.nanstd(arr, ddof=1) if np.sum(valid) > 1 else 0.0

    ucl = m + 3 * s if np.isfinite(m) and np.isfinite(s) else np.nan
    lcl = m - 3 * s if np.isfinite(m) and np.isfinite(s) else np.nan

    beyond = np.zeros(len(arr), dtype=bool)
    if np.isfinite(ucl) and np.isfinite(lcl):
        beyond = (arr > ucl) | (arr < lcl)

    r2 = np.zeros(len(arr), dtype=bool)
    r3 = np.zeros(len(arr), dtype=bool)
    if np.isfinite(m):
        r2 = nelson_rule_2_same_side(arr, mean=m, k=9)
        r3 = nelson_rule_3_trend(arr, k=6)

    return m, s, ucl, lcl, beyond, r2, r3

# ----------------------------
# Sidebar: data source
# ----------------------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Use:", ["Demo data", "Upload CSV"], index=0)

if mode == "Demo data":
    st.sidebar.header("Demo settings")
    n = st.sidebar.slider("Rows", 100, 20000, 800, 100)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
    df_raw = make_demo_data(n=int(n), seed=int(seed))
else:
    st.sidebar.header("Upload")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to continue, or switch to **Demo data**.")
        st.stop()
    df_raw = read_csv(uploaded)

st.subheader("Data preview")
st.dataframe(df_raw.head(50), use_container_width=True)

# ----------------------------
# Column mapping + feature selection
# ----------------------------
st.sidebar.header("Column Mapping")

cols = list(df_raw.columns)
options = ["—"] + cols

def guess_col(possible):
    low = {c.lower(): c for c in cols}
    for name in possible:
        if name in low:
            return low[name]
    return "—"

batch_default = guess_col(["batchid", "batch_id", "batch", "lot", "batchno", "batch_no", "id"])
time_default  = guess_col(["timestamp", "datetime", "date_time", "date", "time", "sampletime", "recordedat"])

map_batch = st.sidebar.selectbox(
    "BatchID (optional)", options,
    index=options.index(batch_default) if batch_default in options else 0
)
map_time = st.sidebar.selectbox(
    "Timestamp (optional but recommended)", options,
    index=options.index(time_default) if time_default in options else 0
)

df = apply_mapping(df_raw, {"BatchID": map_batch, "Timestamp": map_time}).copy()

if "BatchID" not in df.columns:
    df["BatchID"] = [f"B{str(i).zfill(5)}" for i in range(len(df))]

if "Timestamp" in df.columns:
    df["Timestamp"] = safe_to_datetime(df["Timestamp"])

st.sidebar.header("Feature Selection")
candidate_features = [c for c in df.columns if c != "BatchID"]
default_features = [c for c in candidate_features if c != "Timestamp"]  # default exclude timestamp from ML
feature_cols = st.sidebar.multiselect(
    "Use these columns for Anomaly + SPC",
    options=candidate_features,
    default=default_features
)

if not feature_cols:
    st.warning("Pick at least one feature column.")
    st.stop()

# Sort by time if available (helps SPC rules)
if "Timestamp" in df.columns and df["Timestamp"].notna().any():
    df = df.sort_values("Timestamp").reset_index(drop=True)

# ----------------------------
# Anomaly model settings
# ----------------------------
st.sidebar.header("Anomaly Model")
contamination = st.sidebar.slider("Expected anomalies (%)", 1, 30, 10, 1) / 100.0
n_estimators = st.sidebar.slider("IsolationForest trees", 50, 600, 300, 50)
use_robust_spc = st.sidebar.checkbox("Use robust SPC (median/MAD)", value=True)

X = df[feature_cols].copy()

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop"
)

iso = IsolationForest(
    n_estimators=int(n_estimators),
    contamination=float(contamination),
    random_state=42
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", iso)])

with st.spinner("Fitting anomaly model..."):
    pipe.fit(X)

decision = pipe.decision_function(X)     # higher = more normal
pred = pipe.predict(X)                  # 1 normal, -1 anomaly
raw_anom = -decision                    # higher = more anomalous
anom_score = minmax_0_100(raw_anom)

df_out = df.copy()
df_out["Anomaly Score (0-100)"] = np.round(anom_score, 2)
df_out["Anomaly Flag"] = np.where(pred == -1, "⚠️ Anomaly", "Normal")

# ----------------------------
# SPC evaluation across numeric features
# ----------------------------
spc_summary = []
row_spc_flags = np.zeros(len(df_out), dtype=int)

numeric_used = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_out[c])]
for col in numeric_used:
    m, s, ucl, lcl, beyond, r2, r3 = spc_flags_for_series(df_out[col], use_robust=use_robust_spc)

    hits = beyond.astype(int) + r2.astype(int) + r3.astype(int)
    row_spc_flags += (hits > 0).astype(int)

    spc_summary.append({
        "Feature": col,
        "Mean": m,
        "Sigma": s,
        "LCL": lcl,
        "UCL": ucl,
        "Beyond 3σ (count)": int(np.sum(beyond)),
        "Nelson R2 (9 same side) (count)": int(np.sum(r2)),
        "Nelson R3 (6 trend) (count)": int(np.sum(r3)),
    })

df_out["SPC Flag Count"] = row_spc_flags
df_out["SPC Flag"] = np.where(df_out["SPC Flag Count"] > 0, "⚠️ SPC", "")

# Review priority = anomaly + SPC weight
df_out["Review Priority"] = np.round(df_out["Anomaly Score (0-100)"] + 10.0 * df_out["SPC Flag Count"], 2)

# ----------------------------
# Summary KPIs
# ----------------------------
st.divider()
st.subheader("Summary")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(df_out):,}")
k2.metric("Anomalies flagged", f"{(df_out['Anomaly Flag'] != 'Normal').sum():,}")
k3.metric("Rows with any SPC flag", f"{(df_out['SPC Flag Count'] > 0).sum():,}")
k4.metric("Features used", f"{len(feature_cols):,}")

with st.expander("SPC limits + Nelson rule counts (numeric features)"):
    if spc_summary:
        st.dataframe(pd.DataFrame(spc_summary), use_container_width=True)
    else:
        st.info("No numeric features selected, so SPC checks were skipped.")

# ----------------------------
# Batches to review
# ----------------------------
st.divider()
st.subheader("Batches to Review")

top_n = st.slider("Show top N", 10, 200, 30)
review_cols = ["BatchID"]
if "Timestamp" in df_out.columns:
    review_cols.append("Timestamp")
review_cols += ["Anomaly Flag", "Anomaly Score (0-100)", "SPC Flag Count", "Review Priority"]

table = df_out.sort_values(["Review Priority"], ascending=False)
st.dataframe(table[review_cols].head(top_n), use_container_width=True)

csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button("Download scored CSV", data=csv, file_name="scored_anomaly_spc.csv", mime="text/csv")

# ----------------------------
# Drill-down
# ----------------------------
st.divider()
st.subheader("Drill-down")

selected = st.selectbox("Select BatchID", df_out["BatchID"].astype(str).unique())
row = df_out[df_out["BatchID"].astype(str) == str(selected)].iloc[0]

cL, cR = st.columns([1.3, 1])
with cL:
    st.write("**Selected row**")
    st.dataframe(pd.DataFrame([row]), use_container_width=True)

with cR:
    st.metric("Anomaly Score", f"{row['Anomaly Score (0-100)']:.2f}")
    st.metric("SPC Flag Count", int(row["SPC Flag Count"]))
    st.metric("Review Priority", f"{row['Review Priority']:.2f}")
    st.write(f"**Anomaly Flag:** {row['Anomaly Flag']}")
    st.write(f"**SPC Flag:** {row['SPC Flag']}")

# ----------------------------
# Control chart plot
# ----------------------------
st.divider()
st.subheader("SPC Control Chart (numeric feature)")

if numeric_used:
    feat = st.selectbox("Feature", numeric_used, index=0)

    m, s, ucl, lcl, beyond, r2, r3 = spc_flags_for_series(df_out[feat], use_robust=use_robust_spc)

    if "Timestamp" in df_out.columns and df_out["Timestamp"].notna().any():
        x_axis = df_out["Timestamp"]
        x_name = "Timestamp"
    else:
        x_axis = np.arange(len(df_out))
        x_name = "Index"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=df_out[feat], mode="lines+markers", name=feat))

    if np.isfinite(m):
        fig.add_trace(go.Scatter(x=x_axis, y=[m] * len(df_out), mode="lines", name="Mean"))
    if np.isfinite(ucl):
        fig.add_trace(go.Scatter(x=x_axis, y=[ucl] * len(df_out), mode="lines", name="UCL (3σ)"))
    if np.isfinite(lcl):
        fig.add_trace(go.Scatter(x=x_axis, y=[lcl] * len(df_out), mode="lines", name="LCL (3σ)"))

    flag_mask = beyond | r2 | r3
    if flag_mask.any():
        fig.add_trace(go.Scatter(
            x=x_axis[flag_mask] if isinstance(x_axis, pd.Series) else x_axis[flag_mask],
            y=df_out.loc[flag_mask, feat],
            mode="markers",
            name="SPC flagged",
        ))

    fig.update_layout(
        height=450,
        xaxis_title=x_name,
        yaxis_title=feat,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric features selected → control chart not available.")
