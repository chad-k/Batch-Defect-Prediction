# -*- coding: utf-8 -*-
"""
Anomaly Detection + SPC Engine with DYNAMIC CONTAMINATION RATE
- Uses statistical analysis of anomaly scores to auto-detect optimal contamination rate
- Falls back to data-driven heuristics if statistical methods are inconclusive
- Provides explanations and recommendations
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

import plotly.graph_objects as go


# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Anomaly + SPC (Dynamic Contamination)", layout="wide")
st.title("Manufacturing Anomaly Detection + SPC (Smart Contamination Rate)")


# ----------------------------
# DYNAMIC CONTAMINATION DETECTION
# ----------------------------
def estimate_optimal_contamination(anomaly_scores: np.ndarray) -> tuple[float, str]:
    """
    Intelligently estimate optimal contamination rate from anomaly score distribution.
    
    Methods (in order of preference):
    1. **Elbow Method**: Find the "knee" in sorted scores (biggest gap)
    2. **Bimodal Detection**: Detect if scores split into "normal" vs "anomalous" clusters
    3. **Statistical Outliers**: Use Z-score to find natural outlier threshold
    4. **Fallback Heuristic**: Based on group count and data properties
    
    Returns:
        (contamination_rate, explanation)
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    
    if len(scores) < 10:
        return 0.10, "Too few groups (n<10); using default 0.10"
    
    # Method 1: ELBOW METHOD (biggest gap in sorted scores)
    sorted_scores = np.sort(scores)
    gaps = np.diff(sorted_scores)
    
    # Find the largest gap
    largest_gap_idx = np.argmax(gaps)
    largest_gap_value = gaps[largest_gap_idx]
    
    # Calculate threshold as percentile where largest gap occurs
    gap_percentile = (largest_gap_idx + 1) / len(scores)
    
    # Only use elbow if gap is significant (>0.5x median gap)
    median_gap = np.median(gaps)
    if largest_gap_value > 0.5 * median_gap and gap_percentile >= 0.05 and gap_percentile <= 0.30:
        reason = f"Elbow method: major gap detected at {gap_percentile*100:.1f}th percentile (gap={largest_gap_value:.3f})"
        return gap_percentile, reason
    
    # Method 2: BIMODAL DETECTION (scores form two clusters)
    # Check if there's a clear separation between normal and anomalous groups
    # Use the valley in the kernel density estimation concept
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    
    # Look for a threshold around 2-3 standard deviations from mean
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Define "anomalous" as > mean + 1.5*std (captures outlier tail)
    threshold_z = mean_score + 1.5 * std_score
    n_anomalies = np.sum(scores > threshold_z)
    contamination_z = max(0.05, min(0.25, n_anomalies / len(scores)))
    
    reason = f"Z-score method: identified {n_anomalies} anomalies ({contamination_z*100:.1f}%) at threshold {threshold_z:.2f}"
    
    if 0.05 <= contamination_z <= 0.25:
        return contamination_z, reason
    
    # Method 3: STATISTICAL OUTLIER DETECTION (IQR method)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers_iqr = np.sum((scores < lower_bound) | (scores > upper_bound))
    contamination_iqr = max(0.05, min(0.25, n_outliers_iqr / len(scores)))
    
    reason = f"IQR method: identified {n_outliers_iqr} outliers ({contamination_iqr*100:.1f}%)"
    
    if 0.05 <= contamination_iqr <= 0.25:
        return contamination_iqr, reason
    
    # Method 4: FALLBACK HEURISTIC
    # Base on group count: larger datasets tend to have fewer anomalies
    n_groups = len(scores)
    if n_groups < 50:
        fallback = 0.15  # Smaller datasets: more sensitive (15% contamination)
    elif n_groups < 200:
        fallback = 0.10  # Medium datasets: balanced (10%)
    else:
        fallback = 0.08  # Large datasets: conservative (8%)
    
    reason = f"Fallback heuristic: {n_groups} groups → {fallback*100:.0f}% contamination"
    return fallback, reason


# ----------------------------
# HELP / TOOLTIP SECTION
# ----------------------------
with st.expander("Help / How to use this dashboard", expanded=False):
    st.markdown(
        """
### What this app does
This app creates **one summary record per Group** (Group = the column you choose: Batch, Lot, Work Order, etc.).  
It then computes:

**SPC (Control Charts)**
- **Xbar chart:** one point per group = mean of the subgroup measurements
- **S chart:** one point per group = within-group std dev (meaningful when subgroup size n ≥ 2)

**Smart Anomaly Detection**
- Fits **IsolationForest** on group-level features
- **Automatically determines contamination rate** using:
  1. Elbow method (finding natural gaps in scores)
  2. Z-score detection (statistical outlier analysis)
  3. IQR method (robust quartile-based outlier detection)
  4. Fallback heuristic (based on group count)
- Provides explanation for the chosen contamination rate

### Contamination Rate Selection
The app intelligently detects how many groups should be flagged as anomalies by analyzing the distribution of anomaly scores.
Instead of a fixed 10%, it might choose 5%, 8%, 12%, etc. based on what the data tells us.

**Why Dynamic?**
- A dataset with 50 groups might have only 1-2 true anomalies (2%)
- A dataset with 500 groups might have 30-50 true anomalies (6-10%)
- Fixed rates don't adapt to different process behaviors
- AI-driven selection finds the natural "elbow" in the data

### Example
If the sorted anomaly scores show: [0.1, 0.2, 0.25, 0.26, ..., 5.0, 6.2, 7.1, 8.5]
- There's a big gap between 0.26 and 5.0 → The app detects this and says "8% contamination" makes sense
- Instead of blindly using 10%
"""
    )


# ----------------------------
# Helpers (same as before)
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def safe_to_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(s))

def minmax_0_100(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(x)
    return 100.0 * (x - lo) / (hi - lo)

def rule_2_same_side(x: np.ndarray, center: float, k: int = 9) -> np.ndarray:
    side = np.where(x >= center, 1, -1)
    out = np.zeros(len(x), dtype=bool)
    run = 1
    for i in range(1, len(x)):
        run = run + 1 if side[i] == side[i - 1] else 1
        if run >= k:
            out[i] = True
    return out

def rule_3_trend(x: np.ndarray, k: int = 6) -> np.ndarray:
    out = np.zeros(len(x), dtype=bool)
    if len(x) < k:
        return out
    for i in range(k - 1, len(x)):
        w = x[i - k + 1 : i + 1]
        d = np.diff(w)
        if np.all(d > 0) or np.all(d < 0):
            out[i] = True
    return out

def c4(n: int) -> float:
    if n <= 1:
        return float("nan")
    return math.sqrt(2.0 / (n - 1.0)) * math.exp(math.lgamma(n / 2.0) - math.lgamma((n - 1.0) / 2.0))

def b3_b4_from_c4(c4v: float) -> tuple[float, float]:
    if not np.isfinite(c4v) or c4v <= 0:
        return (float("nan"), float("nan"))
    t = math.sqrt(max(0.0, 1.0 - c4v * c4v)) / c4v
    b3 = max(0.0, 1.0 - 3.0 * t)
    b4 = 1.0 + 3.0 * t
    return (b3, b4)

def pooled_within_sigma(ns: np.ndarray, ss: np.ndarray) -> float:
    ns = np.asarray(ns, dtype=float)
    ss = np.asarray(ss, dtype=float)
    dof = np.sum(np.clip(ns - 1.0, 0.0, None))
    if dof <= 0:
        return float("nan")
    num = np.sum((np.clip(ns - 1.0, 0.0, None)) * (ss ** 2))
    return float(math.sqrt(num / dof))

def overall_sigma_from_points(all_points: np.ndarray) -> float:
    all_points = np.asarray(all_points, dtype=float)
    all_points = all_points[np.isfinite(all_points)]
    if len(all_points) < 2:
        return float("nan")
    return float(np.std(all_points, ddof=1))

def capability(mu: float, sigma: float, lsl: float | None, usl: float | None) -> tuple[float, float]:
    if lsl is None or usl is None:
        return (np.nan, np.nan)
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0 and np.isfinite(lsl) and np.isfinite(usl) and usl > lsl):
        return (np.nan, np.nan)
    cp = (usl - lsl) / (6.0 * sigma)
    cpk = min((usl - mu) / (3.0 * sigma), (mu - lsl) / (3.0 * sigma))
    return (cp, cpk)

def extract_group_values_long(df: pd.DataFrame, group_col: str, group_val: str, meas_col: str) -> np.ndarray:
    s = df.loc[df[group_col].astype(str) == str(group_val), meas_col]
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals

def extract_group_values_wide(df: pd.DataFrame, group_col: str, group_val: str, member_cols: list[str]) -> np.ndarray:
    block = df.loc[df[group_col].astype(str) == str(group_val), member_cols]
    if block.empty:
        return np.array([], dtype=float)
    vals = block.to_numpy().ravel()
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals

def agg_numeric_features(
    df: pd.DataFrame,
    gkey: pd.Series,
    num_cols: list[str],
    time_col: str | None,
    include_last: bool,
) -> pd.DataFrame:
    base = pd.DataFrame({"Group": pd.Index(gkey.unique()).astype(str)})
    if not num_cols:
        return base

    gdf = df.copy()
    gdf["_g"] = gkey.astype(str)

    for c in num_cols:
        col = gdf[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        s = pd.to_numeric(col, errors="coerce").astype(float)
        gg = s.groupby(gdf["_g"], dropna=False)
        stats = gg.agg(["mean", "std", "min", "max"])
        stats["std"] = stats["std"].fillna(0.0)
        stats = stats.astype(float)
        stats["range"] = stats["max"] - stats["min"]

        stats = stats.rename(columns={
            "mean": f"{c}__mean",
            "std": f"{c}__std",
            "min": f"{c}__min",
            "max": f"{c}__max",
            "range": f"{c}__range",
        }).reset_index().rename(columns={"_g": "Group"})

        stats["Group"] = stats["Group"].astype(str)
        base = base.merge(stats, on="Group", how="left")

    if include_last and time_col is not None and time_col in gdf.columns and gdf[time_col].notna().any():
        gdf2 = gdf.copy()
        gdf2[time_col] = safe_to_datetime(gdf2[time_col])
        gdf2 = gdf2.sort_values(["_g", time_col])

        for c in num_cols:
            col = gdf2[c]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            s = pd.to_numeric(col, errors="coerce").astype(float)
            last = s.groupby(gdf2["_g"], dropna=False).last()
            base = base.merge(
                last.to_frame(name=f"{c}__last").reset_index().rename(columns={"_g": "Group"}),
                on="Group",
                how="left",
            )

    return base


# ----------------------------
# Sidebar: Data source
# ----------------------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio(
    "Use:",
    ["Upload CSV", "Demo data"],
    index=0,
    help="Upload your real data, or use a demo dataset to test the dashboard."
)

if mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a CSV file. Column names can be anything—you will map them below."
    )
    if not uploaded:
        st.info("Upload a CSV to continue.")
        st.stop()
    df = read_csv(uploaded)
else:
    rng = np.random.default_rng(42)
    demo = []
    for b in range(1, 301):
        batch = f"B{b:05d}"
        n = int(rng.integers(1, 9))
        base = rng.normal(1.500, 0.02) + (0.03 * (b / 300) if b > 200 else 0)
        row_ct = int(rng.integers(1, 4))

        for r in range(row_ct):
            rec = {
                "Timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(b * 30 + r * 2, unit="min"),
                "Batch": batch,
                "Machine": rng.choice(["M1", "M2", "M3"]),
                "Shift": rng.choice(["Day", "Night"]),
                "Temp": float(np.round(rng.normal(200, 3), 2)),
                "Pressure": float(np.round(rng.normal(5.0, 0.4), 3)),
                "Speed": float(np.round(rng.normal(100, 6), 2)),
                "AlarmActive": bool(rng.random() < 0.05),
            }
            for j in range(1, 9):
                rec[f"Sample{j}"] = float(np.round(rng.normal(base, 0.006), 4)) if j <= n else np.nan
            demo.append(rec)

    df = pd.DataFrame(demo)

st.subheader("Raw data preview")
st.dataframe(df.head(50), use_container_width=True)

all_cols = list(df.columns)

# ----------------------------
# Grouping + timestamp
# ----------------------------
st.sidebar.header("Grouping")
group_col = st.sidebar.selectbox(
    "Group by (required)",
    options=all_cols,
    index=0,
    help="This is the column that identifies the thing you want ONE point per group for (Batch, Lot, WorkOrder, etc.)."
)

time_col_choice = st.sidebar.selectbox(
    "Timestamp (optional)",
    options=["—"] + all_cols,
    index=0,
    help="Optional. Used to order groups and (if enabled) compute 'last value' for parameters/traces."
)
time_col = None if time_col_choice == "—" else time_col_choice
if time_col is not None:
    df[time_col] = safe_to_datetime(df[time_col])

gkey = df[group_col].astype(str)

# Order groups
if time_col is not None and df[time_col].notna().any():
    group_order = df.assign(_g=gkey).groupby("_g", dropna=False)[time_col].min()
else:
    row_positions = pd.Series(range(len(df)), index=gkey)
    group_order = row_positions.groupby(level=0, dropna=False).min()

order_map = group_order.to_dict()
groups_sorted = sorted(pd.Index(gkey.unique()).astype(str), key=lambda g: order_map.get(g, 10**18))

# ----------------------------
# Measurement mapping (manual)
# ----------------------------
st.sidebar.header("Subgroup Measurements")
layout = st.sidebar.radio(
    "Measurement layout",
    ["Long (one measurement column)", "Wide (multiple member columns)"],
    index=1,
    help=(
        "Long = one measurement column repeated across multiple rows per group.\n"
        "Wide = multiple member columns (Sample1..SampleN, Cav1..CavN) that form a subgroup per group."
    )
)

if layout == "Long (one measurement column)":
    meas_col = st.sidebar.selectbox(
        "Measurement column",
        options=all_cols,
        index=0,
        help="Select the measurement column for subgroup values. It can be named anything. We'll coerce it to numeric."
    )
    member_cols = []
else:
    member_cols = st.sidebar.multiselect(
        "Member columns (subgroup readings)",
        options=all_cols,
        default=all_cols[:5] if len(all_cols) >= 5 else all_cols,
        help="Select the columns that contain subgroup members (e.g., Sample1..SampleN, Cav1..CavN). We'll coerce them to numeric."
    )
    if not member_cols:
        st.error("Select at least one subgroup member column.")
        st.stop()
    meas_col = None

# ----------------------------
# Machine parameters / trace fields
# ----------------------------
st.sidebar.header("Machine parameters / Trace fields")
st.sidebar.caption("Select numeric-ish process parameters/traces (Temp, Pressure, Speed, Setpoints, Alarm flags, etc.). We'll coerce to numeric.")

exclude = set(member_cols) if member_cols else set()
if meas_col is not None:
    exclude.add(meas_col)
exclude.add(group_col)
if time_col is not None:
    exclude.add(time_col)

param_candidates = [c for c in all_cols if c not in exclude]
param_cols = st.sidebar.multiselect(
    "Parameter/trace columns (coerced to numeric)",
    options=param_candidates,
    default=[],
    help="These will be aggregated per group (mean/std/min/max/range + optional last). Non-numeric values become NaN."
)

include_last = st.sidebar.checkbox(
    "Include last value per group (requires Timestamp)",
    value=False,
    help="Adds param__last per group using the last value by timestamp."
)

# ----------------------------
# Categorical context
# ----------------------------
st.sidebar.header("Categorical context (optional)")
cat_candidates = [c for c in all_cols if c not in {group_col, (time_col or "")}]
context_cats = st.sidebar.multiselect(
    "Categorical context columns",
    options=cat_candidates,
    default=[],
    help="Optional context like Machine/Operator/Shift. We'll use the mode (most common value) within each group."
)

# ----------------------------
# Build subgroup stats per group
# ----------------------------
rows = []
all_points = []

for gv in groups_sorted:
    if layout == "Long (one measurement column)":
        vals = extract_group_values_long(df, group_col, gv, meas_col)
    else:
        vals = extract_group_values_wide(df, group_col, gv, member_cols)

    all_points.append(vals)
    n_i = int(len(vals))

    if n_i == 0:
        xbar = np.nan
        s_i = np.nan
        r_i = np.nan
    elif n_i == 1:
        xbar = float(vals[0])
        s_i = 0.0
        r_i = 0.0
    else:
        xbar = float(np.mean(vals))
        s_i = float(np.std(vals, ddof=1))
        r_i = float(np.max(vals) - np.min(vals))

    rows.append({"Group": str(gv), "n": n_i, "Xbar": xbar, "S": s_i, "R": r_i})

spc = pd.DataFrame(rows)
spc = spc[spc["n"] > 0].copy()
if spc.empty:
    st.error("No groups with valid measurements found.")
    st.stop()

all_points_flat = (
    np.concatenate([a for a in all_points if len(a) > 0])
    if any(len(a) > 0 for a in all_points)
    else np.array([], dtype=float)
)

# ----------------------------
# SPC calculations (Xbar-S charts)
# ----------------------------
Xbarbar = float(np.nanmean(spc["Xbar"].to_numpy(dtype=float)))
sigma_within = pooled_within_sigma(spc["n"].to_numpy(dtype=float), spc["S"].to_numpy(dtype=float))
sigma_overall = overall_sigma_from_points(all_points_flat)

if not np.isfinite(sigma_within) or sigma_within <= 0:
    sigma_within = sigma_overall

n_arr = spc["n"].to_numpy(dtype=float)
spc["UCLx"] = Xbarbar + 3.0 * sigma_within / np.sqrt(n_arr)
spc["LCLx"] = Xbarbar - 3.0 * sigma_within / np.sqrt(n_arr)

x = spc["Xbar"].to_numpy(dtype=float)
flag_r1 = (x > spc["UCLx"].to_numpy(dtype=float)) | (x < spc["LCLx"].to_numpy(dtype=float))
flag_r2 = rule_2_same_side(x, center=Xbarbar, k=9)
flag_r3 = rule_3_trend(x, k=6)

spc["Nelson_R1"] = flag_r1
spc["Nelson_R2"] = flag_r2
spc["Nelson_R3"] = flag_r3
spc["Xbar_Flag"] = flag_r1 | flag_r2 | flag_r3

# S chart
s = spc["S"].to_numpy(dtype=float)
n_int = spc["n"].to_numpy(dtype=int)
mask_n2 = n_int >= 2
Sbar = float(np.nanmean(s[mask_n2])) if np.any(mask_n2) else float("nan")

UCLs = np.full(len(spc), np.nan, dtype=float)
LCLs = np.full(len(spc), np.nan, dtype=float)
flag_s = np.zeros(len(spc), dtype=bool)

if np.isfinite(Sbar) and Sbar >= 0:
    for i, ni in enumerate(n_int):
        if ni >= 2:
            c4v = c4(int(ni))
            b3, b4 = b3_b4_from_c4(c4v)
            if np.isfinite(b3) and np.isfinite(b4):
                LCLs[i] = b3 * Sbar
                UCLs[i] = b4 * Sbar
                flag_s[i] = (s[i] > UCLs[i]) | (s[i] < LCLs[i])

spc["Sbar"] = Sbar
spc["UCLs"] = UCLs
spc["LCLs"] = LCLs
spc["S_Flag"] = flag_s
spc["SPC_Flag"] = spc["Xbar_Flag"] | spc["S_Flag"]

# ----------------------------
# Optional specs
# ----------------------------
st.sidebar.header("Specs (optional)")
use_specs = st.sidebar.checkbox(
    "Enable spec limits (Cp/Cpk/Ppk)",
    value=False,
    help="If you enter LSL/USL, the app will compute Cp/Cpk using within sigma and Pp/Ppk using overall sigma."
)

lsl = usl = None
if use_specs:
    lsl_txt = st.sidebar.text_input("LSL", "", help="Lower Spec Limit (numeric).")
    usl_txt = st.sidebar.text_input("USL", "", help="Upper Spec Limit (numeric).")
    try:
        lsl = float(lsl_txt) if lsl_txt.strip() != "" else None
    except Exception:
        lsl = None
    try:
        usl = float(usl_txt) if usl_txt.strip() != "" else None
    except Exception:
        usl = None

mu = float(np.nanmean(all_points_flat)) if len(all_points_flat) else float("nan")
Cp, Cpk = capability(mu, sigma_within, lsl, usl) if use_specs else (np.nan, np.nan)
Pp, Ppk = capability(mu, sigma_overall, lsl, usl) if use_specs else (np.nan, np.nan)

# ----------------------------
# Build anomaly feature table
# ----------------------------
feat = spc[["Group", "n", "Xbar", "S", "R"]].copy()

num_agg = agg_numeric_features(df=df, gkey=gkey, num_cols=param_cols, time_col=time_col, include_last=include_last)
feat = feat.merge(num_agg, on="Group", how="left")

if context_cats:
    tmp = pd.DataFrame({"Group": pd.Index(gkey.unique()).astype(str)})
    gdf = df.copy()
    gdf["_g"] = gkey.astype(str)
    for c in context_cats:
        mode_series = gdf.groupby("_g", dropna=False)[c].agg(
            lambda s_: s_.mode().iloc[0] if len(s_.mode()) else None
        )
        mode_df = mode_series.to_frame(name=c).reset_index().rename(columns={"_g": "Group"})
        tmp = tmp.merge(mode_df, on="Group", how="left", suffixes=("", f"_{c}"))
    feat = feat.merge(tmp[["Group"] + context_cats], on="Group", how="left", suffixes=("", "_ctx"))

# ----------------------------
# ANOMALY MODEL WITH DYNAMIC CONTAMINATION
# ----------------------------
st.sidebar.header("Anomaly Model (Smart Contamination)")

X_anom = feat.drop(columns=["Group"]).copy()
cat_cols = X_anom.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_anom.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ],
    remainder="drop"
)

iso = IsolationForest(n_estimators=300, contamination=0.10, random_state=42)
pipe = Pipeline(steps=[("prep", preprocess), ("model", iso)])

with st.spinner("Fitting anomaly model..."):
    pipe.fit(X_anom)

decision = pipe.decision_function(X_anom)
pred_initial = pipe.predict(X_anom)
anom_raw = -decision
anom_score_raw = minmax_0_100(anom_raw)

# ✅ ESTIMATE OPTIMAL CONTAMINATION BASED ON SCORES
optimal_contamination, contamination_reason = estimate_optimal_contamination(anom_score_raw)

# Display contamination decision
st.sidebar.info(
    f"**Smart Contamination Decision:**\n\n"
    f"Selected: **{optimal_contamination*100:.1f}%**\n\n"
    f"Reason: {contamination_reason}"
)

# Allow manual override
override_contamination = st.sidebar.checkbox(
    "Override contamination rate?",
    value=False,
    help="Enable this if you want to manually adjust the automatically-detected contamination rate."
)
if override_contamination:
    optimal_contamination = st.sidebar.slider(
        "Manual contamination rate",
        min_value=0.02,
        max_value=0.30,
        value=optimal_contamination,
        step=0.01,
        format="%.2f",
        help="Set the percentage of groups to flag as anomalies (2-30%). Lower = stricter, Higher = more sensitive."
    )

# REFIT with optimal contamination
iso_final = IsolationForest(n_estimators=300, contamination=optimal_contamination, random_state=42)
pipe_final = Pipeline(steps=[("prep", preprocess), ("model", iso_final)])

with st.spinner(f"Refitting with contamination={optimal_contamination*100:.1f}%..."):
    pipe_final.fit(X_anom)

decision_final = pipe_final.decision_function(X_anom)
pred_final = pipe_final.predict(X_anom)
anom_score = minmax_0_100(-decision_final)

feat["Anomaly Score (0-100)"] = np.round(anom_score, 2)
feat["Anomaly Flag"] = np.where(pred_final == -1, "⚠️ Anomaly", "Normal")

# Combine outputs
out = spc.merge(feat[["Group", "Anomaly Score (0-100)", "Anomaly Flag"]], on="Group", how="left")
out["Review Priority"] = np.round(out["Anomaly Score (0-100)"] + 25.0 * out["SPC_Flag"].astype(int), 2)

# ----------------------------
# Summary
# ----------------------------
st.divider()
st.subheader("Summary")

meas_desc = f"Long: `{meas_col}`" if layout == "Long (one measurement column)" else "Wide: " + ", ".join([f"`{c}`" for c in member_cols])
st.write(
    f"""
- **Group key:** `{group_col}`
- **Subgroup mapping:** {meas_desc}
- **Variable subgroup sizes supported (including n=1)**
- **SPC:** Xbar–S with dynamic limits per group (limits scale with √n)
- **Smart Anomaly Detection:** IsolationForest with AI-determined contamination = **{optimal_contamination*100:.1f}%**
"""
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Groups", f"{len(out):,}")
k2.metric("⚠️ Anomaly groups", f"{(out['Anomaly Flag'] != 'Normal').sum():,}")
k3.metric("SPC flagged groups", f"{out['SPC_Flag'].sum():,}")
k4.metric("σ within", f"{sigma_within:.6g}" if np.isfinite(sigma_within) else "NA")
k5.metric("σ overall", f"{sigma_overall:.6g}" if np.isfinite(sigma_overall) else "NA")

if use_specs and (lsl is not None) and (usl is not None):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{mu:.6g}" if np.isfinite(mu) else "NA")
    c2.metric("Cp (within)", f"{Cp:.3f}" if np.isfinite(Cp) else "NA")
    c3.metric("Cpk (within)", f"{Cpk:.3f}" if np.isfinite(Cpk) else "NA")
    c4.metric("Ppk (overall)", f"{Ppk:.3f}" if np.isfinite(Ppk) else "NA")

# ----------------------------
# Anomaly Score Distribution
# ----------------------------
st.divider()
st.subheader("Anomaly Score Distribution (why contamination was chosen)")

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=anom_score, nbinsx=30, name="Anomaly Scores", opacity=0.7))

# Add threshold line
threshold_score = np.percentile(anom_score, (1 - optimal_contamination) * 100)
fig_dist.add_vline(x=threshold_score, line_dash="dash", line_color="red", 
                   annotation_text=f"Threshold ({optimal_contamination*100:.1f}%)", annotation_position="top right")

fig_dist.update_layout(
    height=350,
    xaxis_title="Anomaly Score (0-100)",
    yaxis_title="Number of Groups",
    title="Distribution of Anomaly Scores"
)
st.plotly_chart(fig_dist, use_container_width=True)

st.caption(f"""
**Interpretation:**
- X-axis: How unusual each group is (0=most normal, 100=most anomalous)
- Red dashed line: Threshold for flagging anomalies
- Groups to the RIGHT of the line: Flagged as ⚠️ Anomaly ({out['Anomaly Flag'].str.contains('⚠️').sum()} groups)
- Groups to the LEFT: Normal operation
""")

# ----------------------------
# Review table + download
# ----------------------------
st.divider()
st.subheader("Groups to Review (group-level)")

top_n = st.slider(
    "Show top N groups",
    10, 300, 30,
    help="Shows the highest review priority groups (anomaly score + SPC bump)."
)

review_cols = [
    "Group", "n", "Xbar", "S", "R",
    "Nelson_R1", "Nelson_R2", "Nelson_R3",
    "Xbar_Flag", "S_Flag", "SPC_Flag",
    "Anomaly Flag", "Anomaly Score (0-100)",
    "Review Priority"
]
st.dataframe(out.sort_values("Review Priority", ascending=False)[review_cols].head(top_n), use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download results CSV",
    data=csv_bytes,
    file_name="group_level_anomaly_spc.csv",
    mime="text/csv",
    help="Downloads the group-level table (one row per group) including SPC + anomaly results."
)

# ----------------------------
# Xbar chart
# ----------------------------
st.divider()
st.subheader("Xbar Chart (dynamic limits for variable n)")

x_axis = out["Group"].astype(str).tolist()

fig_x = go.Figure()
fig_x.add_trace(go.Scatter(x=x_axis, y=out["Xbar"], mode="lines+markers", name="Xbar"))
fig_x.add_trace(go.Scatter(x=x_axis, y=[Xbarbar] * len(out), mode="lines", name="X̄̄"))
fig_x.add_trace(go.Scatter(x=x_axis, y=out["UCLx"], mode="lines", name="UCLx"))
fig_x.add_trace(go.Scatter(x=x_axis, y=out["LCLx"], mode="lines", name="LCLx"))

mask_xflag = out["Xbar_Flag"].to_numpy(dtype=bool)
if mask_xflag.any():
    fig_x.add_trace(go.Scatter(
        x=np.array(x_axis)[mask_xflag],
        y=out.loc[mask_xflag, "Xbar"],
        mode="markers",
        name="Xbar flagged",
    ))

fig_x.update_layout(height=420, xaxis_title=f"{group_col} (ordered)", yaxis_title="Xbar (group mean)")
st.plotly_chart(fig_x, use_container_width=True)

# ----------------------------
# S chart
# ----------------------------
st.subheader("S Chart (n≥2 meaningful; n=1 uses S=0)")

fig_s = go.Figure()
fig_s.add_trace(go.Scatter(x=x_axis, y=out["S"], mode="lines+markers", name="S"))
if np.isfinite(Sbar):
    fig_s.add_trace(go.Scatter(x=x_axis, y=[Sbar] * len(out), mode="lines", name="S̄"))
fig_s.add_trace(go.Scatter(x=x_axis, y=out["UCLs"], mode="lines", name="UCLs"))
fig_s.add_trace(go.Scatter(x=x_axis, y=out["LCLs"], mode="lines", name="LCLs"))

mask_sflag = out["S_Flag"].to_numpy(dtype=bool)
if mask_sflag.any():
    fig_s.add_trace(go.Scatter(
        x=np.array(x_axis)[mask_sflag],
        y=out.loc[mask_sflag, "S"],
        mode="markers",
        name="S flagged",
    ))

fig_s.update_layout(height=420, xaxis_title=f"{group_col} (ordered)", yaxis_title="S (within-group std dev)")
st.plotly_chart(fig_s, use_container_width=True)

# ----------------------------
# Drilldown
# ----------------------------
st.divider()
st.subheader("Drill-down: raw rows for a selected group")

chosen = st.selectbox(
    f"Select {group_col}",
    out["Group"].astype(str).unique(),
    help="Pick a group to see raw rows/values that formed the subgroup measurements and parameter traces."
)

left, right = st.columns([1.1, 1])

with left:
    st.write("**Group-level summary**")
    st.dataframe(out[out["Group"].astype(str) == str(chosen)][review_cols], use_container_width=True)

with right:
    st.write("**Raw rows for this group**")
    show_cols = [group_col]
    if time_col is not None:
        show_cols.append(time_col)

    show_cols += context_cats
    if layout == "Long (one measurement column)":
        show_cols.append(meas_col)
    else:
        show_cols += member_cols
    show_cols += param_cols

    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(df[df[group_col].astype(str) == str(chosen)][show_cols], use_container_width=True)
