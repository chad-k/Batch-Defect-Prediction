# -*- coding: utf-8 -*-
"""
Manufacturing Anomaly Detection Engine (Refactored)
- Multivariate anomaly detection on numeric features
- Feature importance analysis (which features drove anomalies?)
- Feature correlation patterns (which combinations = anomalies?)
- Group comparison (anomalous vs normal)
- Part number filtering
- Dynamic contamination rate detection
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Multivariate Anomaly Detection (Process Parameters)", layout="wide")
st.title("Manufacturing Anomaly Detection (Process Parameters)")


# ----------------------------
# DYNAMIC CONTAMINATION DETECTION
# ----------------------------
def estimate_optimal_contamination(anomaly_scores: np.ndarray) -> tuple[float, str]:
    """
    Intelligently estimate optimal contamination rate from anomaly score distribution.
    
    Methods (in order of preference):
    1. **Elbow Method**: Find the "knee" in sorted scores (biggest gap)
    2. **Z-score method**: Statistical outlier detection
    3. **IQR method**: Robust quartile-based detection
    4. **Fallback Heuristic**: Based on group count
    
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
    largest_gap_idx = np.argmax(gaps)
    largest_gap_value = gaps[largest_gap_idx]
    gap_percentile = (largest_gap_idx + 1) / len(scores)
    
    median_gap = np.median(gaps)
    if largest_gap_value > 0.5 * median_gap and gap_percentile >= 0.05 and gap_percentile <= 0.30:
        reason = f"Elbow method: major gap detected at {gap_percentile*100:.1f}th percentile (gap={largest_gap_value:.3f})"
        return gap_percentile, reason
    
    # Method 2: Z-SCORE METHOD
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold_z = mean_score + 1.5 * std_score
    n_anomalies = np.sum(scores > threshold_z)
    contamination_z = max(0.05, min(0.25, n_anomalies / len(scores)))
    
    if 0.05 <= contamination_z <= 0.25:
        reason = f"Z-score method: identified {n_anomalies} anomalies ({contamination_z*100:.1f}%) at threshold {threshold_z:.2f}"
        return contamination_z, reason
    
    # Method 3: IQR METHOD
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    n_outliers_iqr = np.sum(scores > upper_bound)
    contamination_iqr = max(0.05, min(0.25, n_outliers_iqr / len(scores)))
    
    if 0.05 <= contamination_iqr <= 0.25:
        reason = f"IQR method: identified {n_outliers_iqr} outliers ({contamination_iqr*100:.1f}%)"
        return contamination_iqr, reason
    
    # Method 4: FALLBACK HEURISTIC
    n_groups = len(scores)
    if n_groups < 50:
        fallback = 0.15
    elif n_groups < 200:
        fallback = 0.10
    else:
        fallback = 0.08
    
    reason = f"Fallback heuristic: {n_groups} groups → {fallback*100:.0f}% contamination"
    return fallback, reason


# ----------------------------
# FEATURE IMPORTANCE CALCULATION
# ----------------------------
def calculate_feature_importance(pipe: Pipeline, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Calculate which features most influenced anomaly detection.
    For unsupervised models like IsolationForest, use correlation with anomaly scores.
    Returns DataFrame with feature names and importance scores.
    """
    try:
        # Transform data through pipeline
        X_prep = pipe.named_steps['prep'].transform(X)
        
        # Ensure X_prep is 2D
        if len(X_prep.shape) != 2:
            return pd.DataFrame()
        
        # Get anomaly scores
        try:
            anomaly_scores = pipe.named_steps['model'].decision_function(X_prep)
        except:
            return pd.DataFrame()
        
        # Ensure both are numpy arrays and correct shape
        X_prep = np.asarray(X_prep)
        anomaly_scores = np.asarray(anomaly_scores).flatten()
        
        # Check shapes match
        if X_prep.shape[0] != len(anomaly_scores):
            return pd.DataFrame()
        
        n_samples = X_prep.shape[0]
        n_features = X_prep.shape[1]
        
        # Get feature names from preprocessor
        transformed_feature_names = []
        
        # Extract feature names from ColumnTransformer
        try:
            preprocessor = pipe.named_steps['prep']
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                    # OneHotEncoder creates feature names
                    try:
                        cat_names = transformer.get_feature_names_out(columns)
                        transformed_feature_names.extend(cat_names)
                    except:
                        transformed_feature_names.extend([f"{col}_{i}" for i in range(len(columns))])
                elif name == 'num':
                    # Numeric features keep their names
                    transformed_feature_names.extend(columns)
        except:
            # Fallback: generic names
            transformed_feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Ensure we have the right number of feature names
        if len(transformed_feature_names) != n_features:
            transformed_feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        # Calculate correlation of each feature with anomaly scores
        importances = []
        valid_features = 0
        
        for i in range(n_features):
            try:
                # Get feature column
                feature_vals = X_prep[:, i].astype(float)
                
                # Check for valid data
                valid_mask = ~(np.isnan(feature_vals) | np.isnan(anomaly_scores))
                if np.sum(valid_mask) < 2:
                    importances.append(0.0)
                    continue
                
                # Get valid data
                feat_valid = feature_vals[valid_mask]
                score_valid = anomaly_scores[valid_mask]
                
                # Calculate absolute correlation
                if np.std(feat_valid) > 0 and np.std(score_valid) > 0:
                    try:
                        corr = np.abs(np.corrcoef(feat_valid, score_valid)[0, 1])
                        importances.append(corr if not np.isnan(corr) else 0.0)
                        valid_features += 1
                    except:
                        importances.append(0.0)
                else:
                    importances.append(0.0)
            except Exception as e:
                importances.append(0.0)
        
        # If no valid features calculated, return empty
        if valid_features == 0:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            "Feature": transformed_feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)
        
        # Normalize to 0-1 range
        max_imp = importance_df["Importance"].max()
        if max_imp > 0:
            importance_df["Importance"] = importance_df["Importance"] / max_imp
        
        return importance_df
    except Exception as e:
        st.warning(f"Could not calculate feature importance: {str(e)[:100]}")
        return pd.DataFrame()


# ----------------------------
# ANOMALY CONTRIBUTION (per-batch explanation)
# ----------------------------
def calculate_anomaly_contribution(X_scaled: np.ndarray, feature_names: list[str], anomaly_flags: np.ndarray) -> pd.DataFrame:
    """
    For anomalous batches, identify which features were most anomalous (high variance from mean).
    Returns: feature contributions per anomalous batch.
    """
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Calculate z-scores
    z_scores = np.abs(X_scaled)
    
    # For anomalous batches, find top 3 contributors
    contributions = []
    for idx, is_anomaly in enumerate(anomaly_flags):
        if is_anomaly == -1:  # Anomalous
            top_3_features = np.argsort(z_scores[idx])[-3:][::-1]
            top_3_scores = z_scores[idx][top_3_features]
            contributions.append({
                "Batch_Index": idx,
                "Top_Feature_1": feature_names[top_3_features[0]],
                "Top_Feature_1_Score": top_3_scores[0],
                "Top_Feature_2": feature_names[top_3_features[1]] if len(top_3_features) > 1 else None,
                "Top_Feature_2_Score": top_3_scores[1] if len(top_3_features) > 1 else 0,
                "Top_Feature_3": feature_names[top_3_features[2]] if len(top_3_features) > 2 else None,
                "Top_Feature_3_Score": top_3_scores[2] if len(top_3_features) > 2 else 0,
            })
    
    return pd.DataFrame(contributions) if contributions else pd.DataFrame()


# ----------------------------
# PARAMETER CORRELATION WITH ANOMALIES
# ----------------------------
def analyze_parameter_correlation(df: pd.DataFrame, context_cats: list[str], anomaly_flags: np.ndarray) -> dict:
    """
    For categorical context columns (Machine, Shift, Operator), calculate anomaly rate per category.
    Returns: dict of {category: {value: anomaly_rate}}
    """
    results = {}
    
    for cat in context_cats:
        if cat not in df.columns:
            continue
        
        # Convert to string to handle NaN values, fill NaN with "Missing"
        cat_values = df[cat].astype(str).fillna("Missing")
        
        anomaly_rates = {}
        
        # Group manually to avoid categorical null issues
        for group_val in cat_values.unique():
            # Get indices where this category value appears
            indices_list = np.where(cat_values == group_val)[0]
            
            # Get anomalies for this group
            group_anomalies = anomaly_flags[indices_list]
            anomaly_count = np.sum(group_anomalies == -1)
            anomaly_rate = anomaly_count / len(indices_list) if len(indices_list) > 0 else 0
            
            anomaly_rates[str(group_val)] = {
                "anomaly_rate": anomaly_rate,
                "count": len(indices_list),
                "anomaly_count": anomaly_count
            }
        
        results[cat] = anomaly_rates
    
    return results


# ----------------------------
# BATCH COMPARISON (Anomalous vs Normal)
# ----------------------------
def compare_batches(feat: pd.DataFrame, anomaly_flags: np.ndarray, param_cols: list[str]) -> dict:
    """
    Calculate mean/std for anomalous vs normal batches for key parameters.
    Looks for aggregated features (param__mean, param__std, etc.)
    """
    feat_with_flag = feat.copy()
    feat_with_flag["_is_anomaly"] = (anomaly_flags == -1)
    
    comparison = {}
    
    # Try aggregated feature names (param__mean, param__std, param__range, etc.)
    for param in param_cols:
        # Try mean first (most relevant for comparison)
        param_mean_col = f"{param}__mean"
        
        if param_mean_col not in feat.columns:
            # If aggregated version doesn't exist, try raw parameter name
            if param not in feat.columns:
                continue
            param_mean_col = param
        
        anomalous_vals = feat_with_flag[feat_with_flag["_is_anomaly"]][param_mean_col].dropna()
        normal_vals = feat_with_flag[~feat_with_flag["_is_anomaly"]][param_mean_col].dropna()
        
        if len(anomalous_vals) > 0 and len(normal_vals) > 0:
            comparison[param] = {
                "anomalous_mean": float(np.mean(anomalous_vals)),
                "anomalous_std": float(np.std(anomalous_vals)),
                "anomalous_count": len(anomalous_vals),
                "normal_mean": float(np.mean(normal_vals)),
                "normal_std": float(np.std(normal_vals)),
                "normal_count": len(normal_vals),
                "difference": float(np.mean(anomalous_vals) - np.mean(normal_vals)),
            }
    
    return comparison


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
# HELP / TOOLTIP SECTION
# ----------------------------
with st.expander("Help / How to use this dashboard", expanded=False):
    st.markdown(
        """
### What this app does
This app analyzes **numeric features per group** (Batch, Lot, Work Order, etc.) to detect groups with unusual feature combinations.

**Key Analyses:**
- **Multivariate Anomaly Detection**: Finds groups with unusual *combinations* of features (measurements, parameters, traces, etc.)
- **Feature Importance**: Which features most often indicate anomalies?
- **Feature Contribution**: For each flagged group, which features made it anomalous?
- **Categorical Correlation**: Which machines/shifts/operators have higher anomaly rates?
- **Group Comparison**: How do anomalous groups differ from normal ones?

### Use Cases
- **Process Monitoring**: Detect batches with unusual process conditions (Temp + Pressure combos)
- **Measurement Analysis**: Find batches with unusual measurement patterns
- **Cross-feature**: Analyze measurements AND parameters together
- **Mislabeling Detection**: Detect groups that might have wrong product type

**Example:**
- "Batch B001 anomalous: Temp=215°C + Pressure=5.8bar (unusual combo for this batch)"
- "Batch B002 measurements unusual: Length consistently low, Width high (possible product mislabel)"

### ⚠️ CRITICAL: Part Number Filtering
Different part numbers have **different process parameters and tolerances**.

**How to use:**
1. Look for "Product Filter" in the sidebar
2. Select your Part Number/SKU column
3. Choose ONE part number to analyze
4. All analysis is now for that part type only

### Contamination Rate Selection
The app intelligently detects how many batches should be flagged as anomalies (typically 5-15%) based on the distribution of anomaly scores.
"""
    )


# ----------------------------
# Sidebar: Data source
# ----------------------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio(
    "Use:",
    ["Upload CSV", "Demo data"],
    index=0,
)

if mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
    )
    if not uploaded:
        st.info("Upload a CSV to continue.")
        st.stop()
    df = read_csv(uploaded)
else:
    rng = np.random.default_rng(42)
    demo = []
    part_numbers = ["PN-A", "PN-B", "PN-C"]
    
    for b in range(1, 301):
        batch = f"B{b:05d}"
        part_no = part_numbers[b % len(part_numbers)]
        
        if part_no == "PN-A":
            base = rng.normal(1.500, 0.02) + (0.03 * (b / 300) if b > 200 else 0)
        elif part_no == "PN-B":
            base = rng.normal(2.000, 0.03) + (0.04 * (b / 300) if b > 200 else 0)
        else:
            base = rng.normal(1.200, 0.015) + (0.025 * (b / 300) if b > 200 else 0)
        
        n = int(rng.integers(1, 9))
        row_ct = int(rng.integers(1, 4))

        for r in range(row_ct):
            rec = {
                "Timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(b * 30 + r * 2, unit="min"),
                "PartNo": part_no,
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
# Part Number Filter
# ----------------------------
st.sidebar.header("Product Filter")

part_number_cols = [c for c in all_cols if any(x in c.lower() for x in ['part', 'pn', 'product', 'sku', 'model'])]

if part_number_cols:
    part_col = st.sidebar.selectbox(
        "Part Number/SKU Column",
        options=part_number_cols,
    )
else:
    part_col = st.sidebar.selectbox("Select Part Number/SKU Column", options=all_cols)

unique_parts = sorted(df[part_col].astype(str).unique())

selected_part = st.sidebar.selectbox(
    "Select Part Number to Analyze",
    options=unique_parts,
)

df_filtered = df[df[part_col].astype(str) == str(selected_part)].copy()

if len(df_filtered) == 0:
    st.error(f"No data found for {part_col} = {selected_part}")
    st.stop()

st.sidebar.success(f"✓ Analyzing {len(df_filtered)} rows of {selected_part}")
df = df_filtered
all_cols = list(df.columns)

# ----------------------------
# Grouping + timestamp
# ----------------------------
st.sidebar.header("Grouping")
group_col = st.sidebar.selectbox(
    "Group by (required)",
    options=all_cols,
    index=0,
)

time_col_choice = st.sidebar.selectbox(
    "Timestamp (optional)",
    options=["—"] + all_cols,
    index=0,
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
# Measurement mapping (optional - for reference only)
# ----------------------------
st.sidebar.header("Measurement Reference (optional)")
layout = st.sidebar.radio(
    "Include measurement columns?",
    ["No", "Long format", "Wide format"],
    index=0,
    help="Optional: include measurement data. You can either select measurements directly in the 'Feature columns' section above, or use this for reference/aggregation purposes.",
)

meas_col = None
member_cols = []

if layout == "Long format":
    meas_col = st.sidebar.selectbox(
        "Measurement column",
        options=all_cols,
        index=0,
    )
elif layout == "Wide format":
    member_cols = st.sidebar.multiselect(
        "Member columns",
        options=all_cols,
        default=all_cols[:5] if len(all_cols) >= 5 else all_cols,
    )

# ----------------------------
# Machine parameters / trace fields
# ----------------------------


# Exclude only group and time columns
exclude = set([group_col])
if time_col is not None:
    exclude.add(time_col)

param_candidates = [c for c in all_cols if c not in exclude]
param_cols = st.sidebar.multiselect(
    "Feature columns (required: 3+)",
    options=param_candidates,
    default=[c for c in ["Temp", "Pressure", "Speed"] if c in param_candidates],
    help="Select 3+ numeric features: measurements, parameters, traces, counts, etc. Will be aggregated per group (mean/std/min/max/range). You can include measurement columns directly here!",
)

if len(param_cols) < 3:
    st.error("Select at least 3 features to analyze.")
    st.stop()

include_last = st.sidebar.checkbox(
    "Include last value per group",
    value=False,
)

# ----------------------------
# Categorical context
# ----------------------------
st.sidebar.header("Categorical Context (optional)")
cat_exclude = {group_col, (time_col or "")} | set(member_cols) | ({meas_col} if meas_col else set())
cat_candidates = [c for c in all_cols if c not in cat_exclude]
context_cats = st.sidebar.multiselect(
    "Categorical columns (Machine, Shift, Operator, etc.)",
    options=cat_candidates,
    default=[c for c in ["Machine", "Shift"] if c in cat_candidates],
    help="Optional context like Machine/Operator/Shift for correlation analysis.",
)

# ===== DIAGNOSTIC INFO =====
with st.expander("🔍 Diagnostic Info - Data Quality Check", expanded=False):
    st.write("**Feature columns before conversion:**")
    diag_before = pd.DataFrame({
        "Feature": param_cols,
        "Type": [str(df[col].dtype) for col in param_cols],
        "Non-Null": [df[col].notna().sum() for col in param_cols],
        "Null Count": [df[col].isna().sum() for col in param_cols],
        "Sample": [str(df[col].iloc[0])[:50] for col in param_cols]
    })
    st.dataframe(diag_before, use_container_width=True)
    
    # Check for ANY missing values
    missing_per_col = {col: df[col].isna().sum() for col in param_cols}
    total_missing = sum(missing_per_col.values())
    st.write(f"**Total missing values across all features:** {total_missing}")

# Convert feature columns to numeric for consistency
df_numeric = df.copy()
for col in param_cols:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

with st.expander("🔍 After Numeric Conversion", expanded=False):
    st.write("**After pd.to_numeric():**")
    diag_after = pd.DataFrame({
        "Feature": param_cols,
        "Type": [str(df_numeric[col].dtype) for col in param_cols],
        "Non-Null": [df_numeric[col].notna().sum() for col in param_cols],
        "NaN Count": [df_numeric[col].isna().sum() for col in param_cols],
        "Sample": [str(df_numeric[col].iloc[0])[:50] for col in param_cols]
    })
    st.dataframe(diag_after, use_container_width=True)

# Check data quality after conversion
rows_before = len(df_numeric)
df_numeric = df_numeric.dropna(subset=param_cols)
rows_after = len(df_numeric)
rows_dropped = rows_before - rows_after

if rows_dropped > 0:
    st.warning(f"⚠️ {rows_dropped} rows have missing values in features ({(rows_dropped/rows_before)*100:.1f}% of data)")
else:
    st.success("✓ All features have valid data - no rows dropped")

# ----------------------------
# Build aggregated features per group
# ----------------------------
rows = []

for gv in groups_sorted:
    rows.append({"Group": str(gv)})

feat = pd.DataFrame(rows)

# Use numeric data for aggregation
num_agg = agg_numeric_features(df=df_numeric, gkey=gkey, num_cols=param_cols, time_col=time_col, include_last=include_last)
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
    feat = feat.merge(tmp[["Group"] + context_cats], on="Group", how="left")

# ----------------------------
# ANOMALY MODEL WITH DYNAMIC CONTAMINATION
# ----------------------------
st.sidebar.header("Anomaly Detection")

X_anom = feat.drop(columns=["Group"]).copy()
cat_cols = X_anom.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols_model = [c for c in X_anom.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols_model),
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

# Estimate optimal contamination
optimal_contamination, contamination_reason = estimate_optimal_contamination(anom_score_raw)

st.sidebar.info(
    f"**Smart Contamination Decision:**\n\n"
    f"Selected: **{optimal_contamination*100:.1f}%**\n\n"
    f"Reason: {contamination_reason}"
)

# Refit with optimal contamination
iso_final = IsolationForest(n_estimators=300, contamination=optimal_contamination, random_state=42)
pipe_final = Pipeline(steps=[("prep", preprocess), ("model", iso_final)])

with st.spinner(f"Refitting with contamination={optimal_contamination*100:.1f}%..."):
    pipe_final.fit(X_anom)

decision_final = pipe_final.decision_function(X_anom)
pred_final = pipe_final.predict(X_anom)
anom_score = minmax_0_100(-decision_final)

feat["Anomaly Score (0-100)"] = np.round(anom_score, 2)
feat["Anomaly Flag"] = np.where(pred_final == -1, "⚠️ Anomaly", "Normal")

# Get preprocessed features for importance calculation
X_transformed = pipe_final.named_steps['prep'].fit_transform(X_anom)
feature_names_transformed = []
for name, trans, cols in preprocess.transformers_:
    if name == "cat":
        encoder = trans
        feature_names_transformed.extend(encoder.get_feature_names_out(cols).tolist())
    elif name == "num":
        feature_names_transformed.extend(cols)

# ----------------------------
# Summary
# ----------------------------
st.divider()
st.subheader("Summary")

st.write(
    f"""
- **Group key:** `{group_col}`
- **Features:** {', '.join([f'`{c}`' for c in param_cols])}
- **Categorical Context:** {', '.join([f'`{c}`' for c in context_cats]) if context_cats else 'None'}
- **Anomaly Detection:** IsolationForest with **{optimal_contamination*100:.1f}%** contamination rate
"""
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Groups", f"{len(feat):,}")
k2.metric("⚠️ Anomaly Groups", f"{(feat['Anomaly Flag'] != 'Normal').sum():,}")
k3.metric("Anomaly Rate", f"{(feat['Anomaly Flag'] != 'Normal').sum() / len(feat) * 100:.1f}%")
k4.metric("Contamination Rate", f"{optimal_contamination*100:.1f}%")

# ----------------------------
# Anomaly Score Distribution
# ----------------------------
st.divider()
st.subheader("1. Anomaly Score Distribution")

fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=anom_score, nbinsx=30, name="Anomaly Scores", opacity=0.7))

threshold_score = np.percentile(anom_score, (1 - optimal_contamination) * 100)
fig_dist.add_vline(x=threshold_score, line_dash="dash", line_color="red", 
                   annotation_text=f"Threshold ({optimal_contamination*100:.1f}%)", annotation_position="top right")

fig_dist.update_layout(
    height=400,
    xaxis_title="Anomaly Score (0-100)",
    yaxis_title="Number of Groups",
)
st.plotly_chart(fig_dist, use_container_width=True)

st.caption("Groups to the RIGHT of the red line are flagged as anomalies.")

# ----------------------------
# Feature Importance
# ----------------------------
st.divider()
st.subheader("2. Feature Importance (Overall)")

st.write("Which features are most indicative of anomalies across all groups?")

with st.spinner("Calculating feature importance..."):
    importance_df = calculate_feature_importance(pipe_final, X_anom, num_cols_model + cat_cols)

if not importance_df.empty:
    # Show top features
    top_importance = importance_df.head(10)
    
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=top_importance["Importance"],
        y=top_importance["Feature"],
        orientation="h",
        marker_color="steelblue"
    ))
    fig_imp.update_layout(
        height=400,
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis_autorange="reversed"
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.caption("Top 10 features that best distinguish anomalous groups from normal ones.")
else:
    st.info("Feature importance calculation not available for this configuration.")

# ----------------------------
# Parameter Correlation with Anomalies
# ----------------------------
st.divider()
st.subheader("3. Category Correlation (Anomaly Rates by Group)")

if context_cats:
    st.write("Which machines/shifts/operators have higher anomaly rates?")
    
    correlation_results = analyze_parameter_correlation(feat, context_cats, pred_final)
    
    for cat_name, cat_data in correlation_results.items():
        st.write(f"**{cat_name}**")
        
        cat_df = pd.DataFrame([
            {
                "Value": k,
                "Anomaly Count": v["anomaly_count"],
                "Total Count": v["count"],
                "Anomaly Rate (%)": round(v["anomaly_rate"] * 100, 1)
            }
            for k, v in cat_data.items()
        ]).sort_values("Anomaly Rate (%)", ascending=False)
        
        fig_cat = go.Figure()
        fig_cat.add_trace(go.Bar(
            x=cat_df["Value"],
            y=cat_df["Anomaly Rate (%)"],
            marker_color=["red" if x > optimal_contamination * 100 else "green" for x in cat_df["Anomaly Rate (%)"]],
            text=cat_df["Anomaly Rate (%)"].round(1),
            textposition="auto",
        ))
        fig_cat.update_layout(
            height=350,
            xaxis_title=cat_name,
            yaxis_title="Anomaly Rate (%)",
        )
        st.plotly_chart(fig_cat, use_container_width=True)
        st.dataframe(cat_df, use_container_width=True)
else:
    st.info("No categorical context columns selected. Add Machine, Shift, Operator, or similar for this analysis.")

# ----------------------------
# Batch Comparison
# ----------------------------
st.divider()
st.subheader("4. Group Comparison (Anomalous vs Normal)")

st.write("How do anomalous groups differ from normal ones in their features?")

comparison_data = compare_batches(feat, pred_final, param_cols)

if comparison_data:
    comparison_df = pd.DataFrame([
        {
            "Parameter": param,
            "Anomalous Mean": round(data["anomalous_mean"], 3),
            "Normal Mean": round(data["normal_mean"], 3),
            "Difference": round(data["difference"], 3),
            "Anomalous Std": round(data["anomalous_std"], 3),
            "Normal Std": round(data["normal_std"], 3),
        }
        for param, data in comparison_data.items()
    ]).sort_values("Difference", key=abs, ascending=False)
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=comparison_df["Parameter"],
        y=comparison_df["Anomalous Mean"],
        name="Anomalous",
        marker_color="red",
        opacity=0.7
    ))
    fig_comp.add_trace(go.Bar(
        x=comparison_df["Parameter"],
        y=comparison_df["Normal Mean"],
        name="Normal",
        marker_color="green",
        opacity=0.7
    ))
    fig_comp.update_layout(
        height=400,
        barmode="group",
        xaxis_title="Parameter",
        yaxis_title="Mean Value",
    )
    st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("No numeric features to compare.")

# ----------------------------
# Results Table
# ----------------------------
st.divider()
st.subheader("Results Table (Group-level)")

top_n = st.slider(
    "Show top N groups (by anomaly score)",
    10, 300, 50,
)

review_cols = ["Group", "Anomaly Score (0-100)", "Anomaly Flag"] + param_cols
display_cols = [c for c in review_cols if c in feat.columns]

st.dataframe(
    feat.sort_values("Anomaly Score (0-100)", ascending=False)[display_cols].head(top_n),
    use_container_width=True
)

csv_bytes = feat.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Results CSV",
    data=csv_bytes,
    file_name="anomaly_detection_results.csv",
    mime="text/csv",
)

# ----------------------------
# Drill-down: Select a group
# ----------------------------
st.divider()
st.subheader("Drill-down: Select a Group for Details")

chosen = st.selectbox(
    f"Select {group_col}",
    feat["Group"].astype(str).unique(),
)

left, right = st.columns([1, 1])

with left:
    st.write("**Group Summary**")
    group_row = feat[feat["Group"].astype(str) == str(chosen)]
    st.dataframe(group_row, use_container_width=True)

with right:
    st.write("**Raw Rows for This Group**")
    show_cols = [group_col]
    if time_col is not None:
        show_cols.append(time_col)
    show_cols += context_cats + param_cols
    # Remove duplicates while maintaining order
    show_cols = list(dict.fromkeys([c for c in show_cols if c in df_numeric.columns]))
    
    st.dataframe(df_numeric[df_numeric[group_col].astype(str) == str(chosen)][show_cols], use_container_width=True)
