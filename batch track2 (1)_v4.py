# -*- coding: utf-8 -*-
"""
Anomaly Detection + SPC Engine (Variable n incl. n=1) with MANUAL column selection
+ includes Machine Parameters / numeric trace fields aggregated per group

- Group by: user-selected (Batch/Lot/WO/etc)
- Measurements:
    A) Long: one measurement column, many rows per group
    B) Wide: multiple member columns, one or many rows per group (pooled)
- SPC:
    Variable-n Xbar–S (dynamic limits per subgroup)
    S chart (n>=2)
    Nelson-ish rules on Xbar (R1/R2/R3)
- Capability (optional): Cp/Cpk (within), Pp/Ppk (overall)
- Anomaly detection: IsolationForest on group-level features:
    [Xbar, S, R, n] + aggregated numeric process params + categorical context

IMPORTANT:
- st.set_page_config() must be the FIRST Streamlit command and only called once.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest

import plotly.graph_objects as go


# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Anomaly + SPC (Variable n, manual mapping)", layout="wide")
st.title("Manufacturing Anomaly Detection + SPC (Variable n, manual mapping)")


# ----------------------------
# Helpers
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

def agg_numeric_features(df: pd.DataFrame, group_col: str, gkey: pd.Series, num_cols: list[str], time_col: str | None) -> pd.DataFrame:
    """
    Aggregate numeric process parameters / trace fields per group.
    Returns columns like: col__mean, col__std, col__min, col__max, col__range, col__last (optional).
    """
    if not num_cols:
        return pd.DataFrame({"Group": pd.Index(gkey.unique()).astype(str)})

    # base
    base = pd.DataFrame({"Group": pd.Index(gkey.unique()).astype(str)})
    base = base.copy()

    gdf = df.copy()
    gdf["_g"] = gkey.astype(str)

    # aggregate stats
    for c in num_cols:
        s = pd.to_numeric(gdf[c], errors="coerce")
        gg = gdf.assign(_v=s).groupby("_g", dropna=False)["_v"]
        tmp = pd.DataFrame({
            "Group": gg.mean().index.astype(str),
            f"{c}__mean": gg.mean().values,
            f"{c}__std": gg.std(ddof=1).fillna(0.0).values,
            f"{c}__min": gg.min().values,
            f"{c}__max": gg.max().values,
        })
        tmp[f"{c}__range"] = tmp[f"{c}__max"] - tmp[f"{c}__min"]
        base = base.merge(tmp, on="Group", how="left")

    # last value per group (only if time exists)
    if time_col is not None and time_col in gdf.columns and gdf[time_col].notna().any():
        gdf2 = gdf.copy()
        gdf2[time_col] = safe_to_datetime(gdf2[time_col])
        gdf2 = gdf2.sort_values([ "_g", time_col ])

        for c in num_cols:
            s = pd.to_numeric(gdf2[c], errors="coerce")
            last = gdf2.assign(_v=s).groupby("_g", dropna=False)["_v"].last()
            base = base.merge(last.rename(f"{c}__last").reset_index().rename(columns={"_g": "Group"}), on="Group", how="left")

    return base


# ----------------------------
# Sidebar: Data source
# ----------------------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Use:", ["Upload CSV", "Demo data"], index=0)

if mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV to continue.")
        st.stop()
    df = read_csv(uploaded)
else:
    # simple demo
    rng = np.random.default_rng(42)
    demo = []
    for b in range(1, 301):
        batch = f"B{b:05d}"
        n = int(rng.integers(1, 9))  # variable n incl 1
        base = rng.normal(1.500, 0.02) + (0.03 * (b / 300) if b > 200 else 0)
        row_ct = int(rng.integers(1, 4))  # multiple rows per batch sometimes

        for r in range(row_ct):
            rec = {
                "Timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(b * 30 + r * 2, unit="min"),
                "Batch": batch,
                "Machine": rng.choice(["M1", "M2", "M3"]),
                "Shift": rng.choice(["Day", "Night"]),
                "Temp": float(np.round(rng.normal(200, 3), 2)),
                "Pressure": float(np.round(rng.normal(5.0, 0.4), 3)),
                "Speed": float(np.round(rng.normal(100, 6), 2)),
            }
            # wide member columns (Sample1..Sample8), some null
            for j in range(1, 9):
                rec[f"Sample{j}"] = float(np.round(rng.normal(base, 0.006), 4)) if j <= n else np.nan
            demo.append(rec)

    df = pd.DataFrame(demo)

st.subheader("Raw data preview")
st.dataframe(df.head(50), use_container_width=True)

# ----------------------------
# Grouping + timestamp
# ----------------------------
st.sidebar.header("Grouping")
all_cols = list(df.columns)

group_col = st.sidebar.selectbox("Group by (required) — batch/lot/workorder/etc", options=all_cols, index=0)

time_col_choice = st.sidebar.selectbox("Timestamp (optional, for ordering + last-value features)", options=["—"] + all_cols, index=0)
time_col = None if time_col_choice == "—" else time_col_choice
if time_col is not None:
    df[time_col] = safe_to_datetime(df[time_col])

gkey = df[group_col].astype(str)

# order groups
if time_col is not None and df[time_col].notna().any():
    group_order = df.assign(_g=gkey).groupby("_g", dropna=False)[time_col].min()
else:
    group_order = df.reset_index().assign(_g=gkey).groupby("_g", dropna=False)["index"].min()
order_map = group_order.to_dict()
groups_sorted = sorted(pd.Index(gkey.unique()).astype(str), key=lambda g: order_map.get(g, 10**18))

# ----------------------------
# Measurement mapping (manual)
# ----------------------------
st.sidebar.header("Subgroup Measurements")
layout = st.sidebar.radio("Measurement layout", ["Long (one column)", "Wide (multiple member columns)"], index=1)

numeric_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if not numeric_candidates:
    st.error("No numeric columns detected. Check your CSV types (measurement columns must be numeric).")
    st.stop()

if layout == "Long (one column)":
    meas_col = st.sidebar.selectbox("Measurement column", options=numeric_candidates, index=0)
    member_cols = []
else:
    member_cols = st.sidebar.multiselect(
        "Subgroup member columns (Sample1..SampleN, Cav1..CavN, etc.)",
        options=numeric_candidates,
        default=numeric_candidates[:5]
    )
    if not member_cols:
        st.error("Select at least one subgroup member column.")
        st.stop()
    meas_col = None

# ----------------------------
# Process parameters / trace fields (numeric) for anomaly model
# ----------------------------
st.sidebar.header("Machine parameters / Trace fields")
st.sidebar.caption("Pick numeric columns like Temp/Pressure/Speed/Setpoints/etc. These will be aggregated per group.")

# Recommend defaults: numeric columns not used as subgroup measurements
exclude = set(member_cols)
if meas_col is not None:
    exclude.add(meas_col)

param_candidates = [c for c in numeric_candidates if c not in exclude]
param_cols = st.sidebar.multiselect(
    "Numeric process parameter columns to include",
    options=param_candidates,
    default=param_candidates
)

include_last = st.sidebar.checkbox("Also include last value per group (requires Timestamp)", value=False)

# ----------------------------
# Categorical context (optional)
# ----------------------------
st.sidebar.header("Categorical context (optional)")
cat_candidates = [c for c in df.columns if c not in {group_col, (time_col or "")} and df[c].dtype == "object"]
context_cats = st.sidebar.multiselect("Categorical context columns", options=cat_candidates, default=cat_candidates[:3])

# ----------------------------
# Build subgroup stats per group (pooled subgroup measurement values)
# ----------------------------
rows = []
all_points = []

for gv in groups_sorted:
    if layout == "Long (one column)":
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

all_points_flat = np.concatenate([a for a in all_points if len(a) > 0]) if any(len(a) > 0 for a in all_points) else np.array([], dtype=float)

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
st.sidebar.header("Specs (optional for Cp/Cpk/Ppk)")
use_specs = st.sidebar.checkbox("Enable spec limits", value=False)
lsl = usl = None
if use_specs:
    lsl_txt = st.sidebar.text_input("LSL", "")
    usl_txt = st.sidebar.text_input("USL", "")
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
# Build anomaly feature table (THIS is the key change you asked for)
# Add aggregated machine parameters / numeric traces per group
# ----------------------------
feat = spc[["Group", "n", "Xbar", "S", "R"]].copy()

# numeric param aggregations
param_time_col = time_col if (include_last and time_col is not None) else None
num_agg = agg_numeric_features(df, group_col, gkey, param_cols, param_time_col)
feat = feat.merge(num_agg, on="Group", how="left")

# categorical context mode per group
if context_cats:
    tmp = pd.DataFrame({"Group": pd.Index(gkey.unique()).astype(str)})
    gdf = df.copy()
    gdf["_g"] = gkey.astype(str)
    for c in context_cats:
        mode_series = gdf.groupby("_g", dropna=False)[c].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
        tmp = tmp.merge(mode_series.rename(c).reset_index().rename(columns={"_g": "Group"}), on="Group", how="left")
    feat = feat.merge(tmp, on="Group", how="left")

# ----------------------------
# Anomaly model (IsolationForest)
# ----------------------------
st.sidebar.header("Anomaly Model")

# Fixed contamination rate (same as mislabel detection)
contamination = 0.10

n_estimators = st.sidebar.slider("IsolationForest trees", 50, 600, 300, 50)

X_anom = feat.drop(columns=["Group"]).copy()

cat_cols = X_anom.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_anom.columns if c not in cat_cols]

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
st.subheader("What’s being used (SPC + anomaly)")

meas_desc = f"Long: `{meas_col}`" if layout == "Long (one column)" else "Wide: " + ", ".join([f"`{c}`" for c in member_cols])
st.write(
    f"""
- **Group key:** `{group_col}`
- **Subgroup measurement mapping:** {meas_desc}
- **Variable subgroup sizes supported (including n=1)**
- **SPC:** Xbar–S with dynamic limits per group (UCL/LCL depend on √n)
- **Anomaly features include:**
  - Xbar, S, R, n (per group)
  - Aggregated machine parameters / trace fields: {(", ".join([f"`{c}`" for c in param_cols]) if param_cols else "None")}
  - Context (categorical modes): {(", ".join([f"`{c}`" for c in context_cats]) if context_cats else "None")}
"""
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Groups", f"{len(out):,}")
k2.metric("Anomaly groups", f"{(out['Anomaly Flag'] != 'Normal').sum():,}")
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

top_n = st.slider("Show top N", 10, 300, 30)

review_cols = [
    "Group", "n", "Xbar", "S", "R",
    "Nelson_R1", "Nelson_R2", "Nelson_R3",
    "Xbar_Flag", "S_Flag", "SPC_Flag",
    "Anomaly Flag", "Anomaly Score (0-100)",
    "Review Priority"
]
st.dataframe(out.sort_values("Review Priority", ascending=False)[review_cols].head(top_n), use_container_width=True)

csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download results CSV", data=csv_bytes, file_name="group_level_anomaly_spc.csv", mime="text/csv")

# ----------------------------
# Xbar chart with dynamic limits
# ----------------------------
st.divider()
st.subheader("Xbar Chart (dynamic limits for variable n)")

x_axis = out["Group"].astype(str).tolist()

fig_x = go.Figure()
fig_x.add_trace(go.Scatter(x=x_axis, y=out["Xbar"], mode="lines+markers", name="Xbar"))
fig_x.add_trace(go.Scatter(x=x_axis, y=[Xbarbar]*len(out), mode="lines", name="X̄̄"))
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

fig_x.update_layout(height=420, xaxis_title=f"{group_col} (ordered)", yaxis_title="Xbar")
st.plotly_chart(fig_x, use_container_width=True)

# ----------------------------
# S chart
# ----------------------------
st.subheader("S Chart (n>=2 meaningful; n=1 gets S=0 and no S-limit flags)")

fig_s = go.Figure()
fig_s.add_trace(go.Scatter(x=x_axis, y=out["S"], mode="lines+markers", name="S"))
if np.isfinite(Sbar):
    fig_s.add_trace(go.Scatter(x=x_axis, y=[Sbar]*len(out), mode="lines", name="S̄"))
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

fig_s.update_layout(height=420, xaxis_title=f"{group_col} (ordered)", yaxis_title="S")
st.plotly_chart(fig_s, use_container_width=True)

# ----------------------------
# Drilldown
# ----------------------------
st.divider()
st.subheader("Drill-down: raw rows for a selected group")

chosen = st.selectbox(f"Select {group_col}", out["Group"].astype(str).unique())

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
    if layout == "Long (one column)":
        show_cols.append(meas_col)
    else:
        show_cols += member_cols
    show_cols += param_cols
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[df[group_col].astype(str) == str(chosen)][show_cols], use_container_width=True)

