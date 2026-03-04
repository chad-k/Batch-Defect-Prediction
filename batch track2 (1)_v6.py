# -*- coding: utf-8 -*-
"""
Anomaly Detection + SPC Engine (Variable n incl. n=1) with MANUAL column selection
+ Machine parameters / numeric trace fields aggregated per group
+ Help section + tooltips
+ Fixed IsolationForest settings:
    - contamination = 0.10
    - n_estimators = 300
+ Robust CSV handling for "numeric" fields that may be strings/booleans

IMPORTANT:
- st.set_page_config() must be the FIRST Streamlit command and only called once.

BUG FIXES:
1. Fixed group ordering: Use explicit index instead of reset_index() to avoid column name collisions
2. Added NaN handling in anomaly model features before fitting IsolationForest
3. Added explicit suffix handling for categorical context merges
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer  # ✅ Correct
from sklearn.ensemble import IsolationForest

import plotly.graph_objects as go


# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Anomaly + SPC (Variable n, manual mapping)", layout="wide")
st.title("Manufacturing Anomaly Detection + SPC (Variable n, manual mapping)")


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

**Anomaly detection (group-level)**
- Fits an **IsolationForest** on group-level features:
  - subgroup stats: **n, Xbar, S, R**
  - aggregated machine parameters / numeric trace fields (mean/std/min/max/range, and optional last)
  - optional categorical context (Machine/Shift/Operator/etc.) using group mode (most common value)

### Subgroup measurements (two layouts)
**Long (one column)**  
- You choose a single measurement column (ex: Thickness, Weight, Value).  
- Subgroup size **n = number of valid rows** for that group.

**Wide (multiple member columns)**  
- You choose several member columns (ex: Sample1..SampleN, Cav1..CavN).  
- Subgroup size **n = number of non-null values** across those columns (pooled across all rows for that group).

### Variable subgroup size (including n = 1)
- This app supports variable subgroup sizes and **n = 1**.
- For n = 1: **Xbar is valid**, but **S is not statistically meaningful** (we store S = 0 and do not "judge" it using S limits).

### Control chart limits (high level)
- **Xbar limits are dynamic per group** (because n varies):
  - UCLxᵢ/LCLxᵢ = X̄̄ ± 3·σ_within/√nᵢ  
- **S chart limits** use B3/B4 factors (computed via a c4 approximation) for n ≥ 2.

### Fixed anomaly model disclaimer
This app uses **IsolationForest with fixed settings**:
- **contamination = 0.10** (about the most unusual ~10% of groups will be labeled "⚠️ Anomaly")
- **n_estimators = 300** (fixed, not user-controlled)

This is a **review threshold**, not a defect/OOS rate.
"""
    )


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    # If you ever see "first upload works weird", uncomment the next two lines:
    # file.seek(0)
    # return pd.read_csv(file)
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

# Nelson-ish rules on ordered Xbar series
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

# SPC constants for S-chart via c4 approximation
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
    """
    Robust aggregation of numeric process parameters / trace fields per group.
    Handles:
      - numeric-looking strings
      - boolean columns
      - duplicate column names (df[c] -> DataFrame)
      - all-NaN columns
    """
    base = pd.DataFrame({"Group": pd.Index(gkey.unique()).astype(str)})
    if not num_cols:
        return base

    gdf = df.copy()
    gdf["_g"] = gkey.astype(str)

    for c in num_cols:
        col = gdf[c]
        if isinstance(col, pd.DataFrame):  # duplicate column names
            col = col.iloc[:, 0]

        # Force numeric (booleans become 0/1, strings become NaN if not parseable)
        s = pd.to_numeric(col, errors="coerce").astype(float)

        gg = s.groupby(gdf["_g"], dropna=False)
        stats = gg.agg(["mean", "std", "min", "max"])
        stats["std"] = stats["std"].fillna(0.0)
        stats = stats.astype(float)  # ensures min/max numeric
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

    # Optional: last value per group
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
    # Demo data (wide member columns + process params + categories)
    rng = np.random.default_rng(42)
    demo = []
    for b in range(1, 301):
        batch = f"B{b:05d}"
        n = int(rng.integers(1, 9))  # variable n incl 1
        base = rng.normal(1.500, 0.02) + (0.03 * (b / 300) if b > 200 else 0)
        row_ct = int(rng.integers(1, 4))  # sometimes multiple rows per batch

        for r in range(row_ct):
            rec = {
                "Timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(b * 30 + r * 2, unit="min"),
                "Batch": batch,
                "Machine": rng.choice(["M1", "M2", "M3"]),
                "Shift": rng.choice(["Day", "Night"]),
                "Temp": float(np.round(rng.normal(200, 3), 2)),
                "Pressure": float(np.round(rng.normal(5.0, 0.4), 3)),
                "Speed": float(np.round(rng.normal(100, 6), 2)),
                "AlarmActive": bool(rng.random() < 0.05),  # boolean trace example
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
# BUG FIX #1: Avoid reset_index() which can cause column name collisions
# Use explicit row indices instead
if time_col is not None and df[time_col].notna().any():
    group_order = df.assign(_g=gkey).groupby("_g", dropna=False)[time_col].min()
else:
    # Create explicit position index without reset_index()
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

# Candidates: all columns (not auto-detecting by dtype for measurement; user picks)
# But we still need to provide choices. We'll show ALL columns; conversion happens later.
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
# Machine parameters / trace fields for anomaly model
# ----------------------------
st.sidebar.header("Machine parameters / Trace fields")
st.sidebar.caption("Select numeric-ish process parameters/traces (Temp, Pressure, Speed, Setpoints, Alarm flags, etc.). We'll coerce to numeric.")

exclude = set(member_cols)
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
# Categorical context (optional)
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
# Build subgroup stats per group (pooled subgroup measurement values)
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
    st.error("No groups with valid measurements found. Check your measurement mapping and data.")
    st.stop()

all_points_flat = (
    np.concatenate([a for a in all_points if len(a) > 0])
    if any(len(a) > 0 for a in all_points)
    else np.array([], dtype=float)
)

# ----------------------------
# Variable-n Xbar–S limits
# ----------------------------
Xbarbar = float(np.nanmean(spc["Xbar"].to_numpy(dtype=float)))

sigma_within = pooled_within_sigma(spc["n"].to_numpy(dtype=float), spc["S"].to_numpy(dtype=float))
sigma_overall = overall_sigma_from_points(all_points_flat)

# fallback if all n=1 (no within variation)
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

# ----------------------------
# S chart (n>=2 only)
# ----------------------------
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
# Optional specs (capability)
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

num_agg = agg_numeric_features(
    df=df,
    gkey=gkey,
    num_cols=param_cols,
    time_col=time_col,
    include_last=include_last
)
feat = feat.merge(num_agg, on="Group", how="left")

# categorical context mode per group (as object/category)
# BUG FIX #3: Add explicit suffix handling for categorical context merges
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
# Anomaly model (IsolationForest) - FIXED settings
# ----------------------------
st.sidebar.header("Anomaly Model")
st.sidebar.info(
    "Fixed IsolationForest settings:\n"
    "- contamination = 0.10 (about ~10% of groups labeled as ⚠️ Anomaly)\n"
    "- n_estimators = 300\n\n"
    "This is a review threshold, not a defect rate."
)

contamination = 0.10
n_estimators = 300

X_anom = feat.drop(columns=["Group"]).copy()

cat_cols = X_anom.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_anom.columns if c not in cat_cols]

# BUG FIX #2: Add NaN handling for numeric features before anomaly model fitting
# Use SimpleImputer to replace NaN with median (or mean) of each numeric column
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ],
    remainder="drop"
)

iso = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    random_state=42,
)

pipe = Pipeline(steps=[("prep", preprocess), ("model", iso)])

with st.spinner("Fitting anomaly model at group level..."):
    pipe.fit(X_anom)

decision = pipe.decision_function(X_anom)  # higher = more normal
pred = pipe.predict(X_anom)               # 1 normal, -1 anomaly
anom_raw = -decision
anom_score = minmax_0_100(anom_raw)

feat["Anomaly Score (0-100)"] = np.round(anom_score, 2)
feat["Anomaly Flag"] = np.where(pred == -1, "⚠️ Anomaly", "Normal")

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
- **Anomaly detection:** IsolationForest on group features + selected parameters/traces (fixed contamination=0.10, trees=300)
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
# Xbar chart with dynamic limits
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

fig_x.update_layout(
    height=420,
    xaxis_title=f"{group_col} (ordered)",
    yaxis_title="Xbar (group mean)"
)
st.plotly_chart(fig_x, use_container_width=True)

# ----------------------------
# S chart
# ----------------------------
st.subheader("S Chart (n≥2 meaningful; n=1 uses S=0 and is not flagged by S limits)")

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

fig_s.update_layout(
    height=420,
    xaxis_title=f"{group_col} (ordered)",
    yaxis_title="S (within-group std dev)"
)
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

    # context, measurement cols, param cols
    show_cols += context_cats
    if layout == "Long (one measurement column)":
        show_cols.append(meas_col)
    else:
        show_cols += member_cols
    show_cols += param_cols

    # keep only existing columns, preserve order
    show_cols = [c for c in show_cols if c in df.columns]

    st.dataframe(df[df[group_col].astype(str) == str(chosen)][show_cols], use_container_width=True)
