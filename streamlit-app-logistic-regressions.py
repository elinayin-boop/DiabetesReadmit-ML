# =============================================================================
# app.py  -  Hospital Readmission Prediction Dashboard (Logistic Regression)
#
# Pipeline order:
#   diabetic_cleaned.csv
#       ↓  load_and_prepare()   - reads CSV, encodes features, builds X and y
#       ↓  build_eda_tables()   - aggregates chart-ready tables from raw data
#       ↓  train_lr_model()     - fits model, runs eval, returns results dict
#       ↓  render_*()           - Streamlit pages draw every chart from results
#
# ── Google Colab setup 
#   Cell 1:
#       !pip install -q streamlit imbalanced-learn shap plotly pyngrok
#
#   Cell 2 (after uploading app.py + diabetic_cleaned.csv to /content/):
#       from pyngrok import ngrok, conf
#       import subprocess, threading, time
#       ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # get free token at ngrok.com
#       threading.Thread(
#           target=lambda: subprocess.run(
#               ["streamlit", "run", "/content/app.py",
#                "--server.port", "8501", "--server.headless", "true"]
#           ), daemon=True
#       ).start()
#       time.sleep(4)
#       print("Open →", ngrok.connect(8501))
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# set_page_config MUST be the first Streamlit call in the script 
st.set_page_config(
    page_title="Hospital Readmission - Logistic Regression Dashboard",
    page_icon="🏥",
    layout="wide",
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Constants 
RANDOM_STATE      = 42
DEFAULT_THRESHOLD = 0.35
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "diabetic_cleaned.csv")


# STEP 1 - load_and_prepare()
# Reads diabetic_cleaned.csv, identifies the target column, one-hot encodes categorical features, and returns a clean numeric feature matrix X and binary target series y.

@st.cache_data(show_spinner=False)
def load_and_prepare(csv_path: str):
    df   = pd.read_csv(csv_path)
    data = df.copy()

    # Identify target column
    target_col = next(
        (c for c in ["readmitted_binary", "target", "label", "readmitted"]
         if c in data.columns), None
    )
    if target_col is None:
        raise ValueError(
            "Could not find a target column. Expected one of: "
            "'readmitted_binary', 'target', 'label', or 'readmitted'."
        )
    if target_col == "readmitted":
        data["readmitted_binary"] = (data["readmitted"] == "<30").astype(int)
        target_col = "readmitted_binary"

    # Dataset-level summary
    positive_rate = float(data[target_col].mean())
    summary_df = pd.DataFrame({
        "metric": ["rows", "columns", "positive_class_count",
                   "negative_class_count", "positive_class_rate"],
        "value":  [int(data.shape[0]), int(data.shape[1]),
                   int((data[target_col] == 1).sum()),
                   int((data[target_col] == 0).sum()),
                   round(positive_rate, 4)],
    })

    # Helper label column used by EDA charts
    data["_readmit_label"] = data[target_col].map(
        {0: "No early readmission", 1: "Readmitted <30 days"}
    )

    # Numeric age mapping (if not already present)
    if "age_numeric" not in data.columns and "age" in data.columns:
        age_map = {
            "[0-10)": 5,   "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
            "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
            "[80-90)": 85, "[90-100)": 95,
        }
        data["age_numeric"] = data["age"].map(age_map)

    # Build feature matrix X
    X = data.drop(columns=[target_col, "_readmit_label"], errors="ignore").copy()
    y = data[target_col].copy()

    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        X = pd.get_dummies(X, columns=object_cols, drop_first=False)

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    return data, X, y, summary_df, target_col
  
# STEP 2 - build_eda_tables

@st.cache_data(show_spinner=False)
def build_eda_tables(data: pd.DataFrame):
    # Class distribution
    readmit_counts = (
        data["_readmit_label"].value_counts()
        .rename_axis("readmitted").reset_index(name="count")
    )

    # Age × readmission grouped bar
    age_readmit = pd.DataFrame(columns=["age_group", "readmitted", "count"])
    if "age_numeric" in data.columns:
        age_bins = pd.cut(data["age_numeric"],
                          bins=[0,10,20,30,40,50,60,70,80,90,100], right=False)
        age_readmit = (
            pd.DataFrame({"age_group": age_bins.astype(str),
                          "readmitted": data["_readmit_label"]})
            .groupby(["age_group", "readmitted"], as_index=False)
            .size().rename(columns={"size": "count"})
        )

    # Box-plot tables
    inpatient_box = (
        data[["number_inpatient", "_readmit_label"]].rename(
            columns={"_readmit_label": "readmitted"})
        if "number_inpatient" in data.columns
        else pd.DataFrame(columns=["number_inpatient", "readmitted"])
    )
    med_box = (
        data[["num_medications", "_readmit_label"]].rename(
            columns={"_readmit_label": "readmitted"})
        if "num_medications" in data.columns
        else pd.DataFrame(columns=["num_medications", "readmitted"])
    )

    # Correlation heatmap
    heatmap_cols = [c for c in
                    ["number_inpatient", "num_medications", "time_in_hospital",
                     "number_diagnoses", "age_numeric",
                     "prior_utilization", "med_change_count"]
                    if c in data.columns]
    corr_df = data[heatmap_cols].corr() if heatmap_cols else pd.DataFrame()

    return {
        "readmit_counts": readmit_counts,
        "age_readmit":    age_readmit,
        "inpatient_box":  inpatient_box,
        "med_box":        med_box,
        "corr_df":        corr_df,
    }

# STEP 3 - train_lr_model()

@st.cache_data(show_spinner=False)
def train_lr_model(X: pd.DataFrame, y: pd.Series):
    # Train / validation / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=RANDOM_STATE, stratify=y_train)

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE on training set only
    X_train_sm, y_train_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train_sc, y_train)

    # Fit Logistic Regression
    lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000,
                            class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train_sm, y_train_sm)

    # Threshold comparison on validation set
    y_val_proba = lr.predict_proba(X_val_sc)[:, 1]
    threshold_df = pd.DataFrame([
        {"threshold": t,
         "recall":    recall_score(y_val, (y_val_proba >= t).astype(int)),
         "precision": precision_score(y_val, (y_val_proba >= t).astype(int), zero_division=0),
         "f1":        f1_score(y_val, (y_val_proba >= t).astype(int), zero_division=0)}
        for t in [0.10, 0.25, 0.35, 0.50]
    ])

    # Test-set evaluation
    y_test_proba = lr.predict_proba(X_test_sc)[:, 1]
    y_test_pred  = (y_test_proba >= DEFAULT_THRESHOLD).astype(int)

    metrics_df = pd.DataFrame({
        "metric": ["AUC-ROC", "F1 (macro)", "F1 (positive)", "Recall", "Precision",
                   "Accuracy", "Balanced Accuracy", "PR-AUC", "Brier Score"],
        "value":  [round(v, 4) for v in [
            roc_auc_score(y_test, y_test_proba),
            f1_score(y_test, y_test_pred, average="macro"),
            f1_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, zero_division=0),
            accuracy_score(y_test, y_test_pred),
            balanced_accuracy_score(y_test, y_test_pred),
            average_precision_score(y_test, y_test_proba),
            brier_score_loss(y_test, y_test_proba),
        ]],
    })

    cm                         = confusion_matrix(y_test, y_test_pred)
    fpr, tpr, _                = roc_curve(y_test, y_test_proba)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_test_proba)

    # 5-fold CV - scores computed from actual data (not hardcoded)
    cv_scores = cross_val_score(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                            max_iter=1000, random_state=RANDOM_STATE))]),
        X_train, y_train, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    )
    cv_df = pd.DataFrame({
        "metric": ["cv_auc_mean", "cv_auc_std", "model_stability",
                   "fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
        "value":  [round(cv_scores.mean(), 4), round(cv_scores.std(), 4),
                   "Stable" if cv_scores.std() < 0.01 else "Unstable",
                   *[round(s, 4) for s in cv_scores]],
    })

    # Coefficients (top 10 positive + top 10 negative)
    coef_series = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
    coef_df     = pd.concat([coef_series.head(10),
                              coef_series.tail(10).sort_values()]).reset_index()
    coef_df.columns = ["feature", "coefficient"]

    # SHAP (optional)
    shap_df = None
    if SHAP_AVAILABLE:
        try:
            X_tr_df = pd.DataFrame(X_train_sc, columns=X.columns)
            X_te_df = pd.DataFrame(X_test_sc,  columns=X.columns)
            sv      = shap.Explainer(lr, X_tr_df)(X_te_df.iloc[:400])
            shap_df = (pd.DataFrame({"feature": X.columns,
                                     "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
                       .sort_values("mean_abs_shap", ascending=False).head(15))
        except Exception:
            shap_df = None

    # Split statistics
    split_df = pd.DataFrame({
        "metric": ["train_rows", "validation_rows", "test_rows",
                   "train_positive_rate", "test_positive_rate",
                   "class_0_after_smote", "class_1_after_smote"],
        "value":  [int(X_train.shape[0]), int(X_val.shape[0]), int(X_test.shape[0]),
                   round(float(y_train.mean()), 4), round(float(y_test.mean()), 4),
                   int((y_train_sm == 0).sum()), int((y_train_sm == 1).sum())],
    })

    return {
        "threshold_df":     threshold_df,
        "metrics_df":       metrics_df,
        "cv_df":            cv_df,
        "coef_df":          coef_df,
        "shap_df":          shap_df,
        "confusion_matrix": cm,
        "roc":              (fpr, tpr),
        "pr":               (pr_recall, pr_precision),
        "split_df":         split_df,
    }

# STEP 4 - Plot helpers

def plot_readmission_distribution(df):
    fig = px.bar(df, x="readmitted", y="count", text="count",
                 title="Readmission Class Distribution")
    fig.update_traces(textposition="outside")
    return fig

def plot_age_readmission(df):
    return px.bar(df, x="age_group", y="count", color="readmitted",
                  barmode="group", title="Age Group vs Readmission")

def plot_box(df, x_col, y_col, title):
    return px.box(df, x=x_col, y=y_col, color=x_col, title=title, points=False)

def plot_heatmap(corr_df):
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.index,
        text=np.round(corr_df.values, 2), texttemplate="%{text}",
        colorscale="RdBu", zmid=0,
    ))
    fig.update_layout(title="Correlation Heatmap of Key Numeric Features")
    return fig

def plot_confusion_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    fig = go.Figure(data=go.Heatmap(
        z=np.array([[tn, fp], [fn, tp]]),
        x=["Not<br>Readmitted", "Readmitted<br>&lt;30d"],
        y=["Not<br>Readmitted", "Readmitted<br>&lt;30d"],
        text=np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]),
        texttemplate="%{text}", colorscale="Blues", showscale=True,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Confusion Matrix (threshold=0.35)",
                      xaxis_title="Predicted label", yaxis_title="True label",
                      margin=dict(l=20, r=20, t=60, b=20))
    fig.update_yaxes(autorange="reversed")
    return fig

def plot_roc_curve(roc_tuple):
    fpr, tpr = roc_tuple
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Logistic Regression"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def plot_pr_curve(pr_tuple):
    recall_vals, precision_vals = pr_tuple
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall_vals, y=precision_vals,
                             mode="lines", name="Logistic Regression"))
    fig.update_layout(title="Precision-Recall Curve",
                      xaxis_title="Recall", yaxis_title="Precision")
    return fig

def plot_threshold_comparison(threshold_df):
    fig = go.Figure()
    for metric in ["recall", "precision", "f1"]:
        fig.add_trace(go.Scatter(x=threshold_df["threshold"], y=threshold_df[metric],
                                 mode="lines+markers", name=metric.capitalize()))
    fig.add_vline(x=0.35, line_dash="dash",
                  annotation_text="Chosen: 0.35", annotation_position="top")
    fig.update_layout(title="Threshold Comparison",
                      xaxis_title="Threshold", yaxis_title="Score",
                      xaxis=dict(tickmode="array",
                                 tickvals=threshold_df["threshold"].tolist()),
                      yaxis=dict(range=[0, 1.05]), legend_title="Metric",
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_cv_auc_chart(cv_df):
    cv_mean     = float(cv_df.loc[cv_df["metric"] == "cv_auc_mean", "value"].iloc[0])
    fold_scores = [
        float(cv_df.loc[cv_df["metric"] == f"fold_{i}", "value"].iloc[0])
        for i in range(1, 6)
    ]
    plot_df = pd.DataFrame({"Fold": list(range(1, 6)), "ROC-AUC": fold_scores})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["Fold"], y=plot_df["ROC-AUC"],
                             mode="lines+markers",
                             hovertemplate="Fold %{x}<br>AUC=%{y:.4f}<extra></extra>"))
    fig.add_hline(y=cv_mean, line_dash="dash",
                  annotation_text=f"Mean = {cv_mean:.4f}",
                  annotation_position="top left")
    fig.update_layout(title="5-Fold CV ROC-AUC",
                      xaxis_title="Fold", yaxis_title="ROC-AUC",
                      xaxis=dict(tickmode="array", tickvals=[1,2,3,4,5]),
                      yaxis=dict(range=[max(0, min(fold_scores)-0.02),
                                        min(1, max(fold_scores)+0.02)]),
                      margin=dict(l=20, r=20, t=45, b=20),
                      height=300, showlegend=False)
    return fig

def plot_coefficients(coef_df):
    return px.bar(coef_df.sort_values("coefficient"),
                  x="coefficient", y="feature", orientation="h",
                  color="coefficient", title="Top Logistic Regression Coefficients")

def plot_shap_bar(shap_df):
    return px.bar(shap_df.sort_values("mean_abs_shap"),
                  x="mean_abs_shap", y="feature", orientation="h",
                  title="Top Mean |SHAP| Features")


# STEP 5 - render_*()

def render_overview(summary_df, split_df):
    st.header("Project Overview")
    st.write(
        "This dashboard presents the **Logistic Regression** track using "
        "**diabetic_cleaned.csv**. Goal: predict whether a patient will be "
        "**readmitted within 30 days**."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows",
            f"{int(summary_df.loc[summary_df['metric']=='rows','value'].iloc[0]):,}")
    with col2:
        st.metric("Columns",
            f"{int(summary_df.loc[summary_df['metric']=='columns','value'].iloc[0])}")
    with col3:
        rate = summary_df.loc[summary_df['metric']=='positive_class_rate','value'].iloc[0]
        st.metric("Positive Rate", f"{rate:.2%}")

    st.info("Target: `readmitted_binary = 1` means readmission within 30 days.")
    left, right = st.columns(2)
    with left:
        st.subheader("Dataset Summary")
        st.dataframe(summary_df, use_container_width=True)
    with right:
        st.subheader("Train / Validation / Test Summary")
        st.dataframe(split_df, use_container_width=True)


def render_eda(eda_views):
    st.header("EDA Insights")

    st.subheader("1. Class Imbalance")
    st.plotly_chart(plot_readmission_distribution(eda_views["readmit_counts"]),
                    use_container_width=True)
    st.caption("Early readmission is the minority class — motivates SMOTE.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("2. Prior Inpatient Visits")
        if not eda_views["inpatient_box"].empty:
            st.plotly_chart(
                plot_box(eda_views["inpatient_box"], "readmitted",
                         "number_inpatient", "Prior Inpatient Visits vs Readmission"),
                use_container_width=True)
    with col2:
        st.subheader("3. Number of Medications")
        if not eda_views["med_box"].empty:
            st.plotly_chart(
                plot_box(eda_views["med_box"], "readmitted",
                         "num_medications", "Number of Medications vs Readmission"),
                use_container_width=True)

    if not eda_views["age_readmit"].empty:
        st.subheader("4. Age Distribution")
        st.plotly_chart(plot_age_readmission(eda_views["age_readmit"]),
                        use_container_width=True)

    if not eda_views["corr_df"].empty:
        st.subheader("5. Correlation Heatmap")
        st.plotly_chart(plot_heatmap(eda_views["corr_df"]), use_container_width=True)


def render_results(outputs):
    st.header("Logistic Regression Results")

    st.subheader("1. Threshold Comparison")
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.plotly_chart(plot_threshold_comparison(outputs["threshold_df"]),
                        use_container_width=True)
    with col_r:
        st.dataframe(outputs["threshold_df"].style.format(
            {"threshold": "{:.2f}", "recall": "{:.3f}",
             "precision": "{:.3f}", "f1": "{:.3f}"}),
            use_container_width=True)
    st.caption("Threshold 0.35 chosen: keeps recall high while improving precision.")

    col1, col2 = st.columns([1, 1.1])

    with col1:
        st.subheader("2. Metrics Summary")
        st.dataframe(
            outputs["metrics_df"].style.format({"value": "{:.4f}"}),
            use_container_width=True
        )

    with col2:
        st.subheader("3. Confusion Matrix")
        st.plotly_chart(
            plot_confusion_matrix(outputs["confusion_matrix"]),
            use_container_width=True
        )

    st.subheader("4. Cross-Validation (5-Fold)")
    cv_display = outputs["cv_df"][outputs["cv_df"]["metric"].isin(
        ["cv_auc_mean", "cv_auc_std", "model_stability"]
    )]

    cv_left, cv_right = st.columns([0.9, 1.6])

    with cv_left:
        st.dataframe(cv_display, use_container_width=True)

    with cv_right:
        st.plotly_chart(
            plot_cv_auc_chart(outputs["cv_df"]),
            use_container_width=True
        )

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("5. ROC Curve")
        st.plotly_chart(plot_roc_curve(outputs["roc"]), use_container_width=True)
    with col4:
        st.subheader("6. Precision-Recall Curve")
        st.plotly_chart(plot_pr_curve(outputs["pr"]), use_container_width=True)


def render_interpretation(outputs):
    st.header("Interpretation")

    st.subheader("1. Top LR Coefficients")
    st.plotly_chart(plot_coefficients(outputs["coef_df"]), use_container_width=True)
    st.dataframe(outputs["coef_df"].style.format({"coefficient": "{:.4f}"}),
                 use_container_width=True)

    if outputs["shap_df"] is not None and not outputs["shap_df"].empty:
        st.subheader("2. SHAP - Mean |contribution| per feature")
        st.plotly_chart(plot_shap_bar(outputs["shap_df"]), use_container_width=True)
    else:
        st.info("SHAP unavailable. Explainability provided via LR coefficients above.")


def render_methodology():
    st.header("Methodology Notes")
    st.markdown(
        "- **Data:** `diabetic_cleaned.csv` — loaded dynamically; model results are computed from the current dataset\n"
        "- **Model:** Logistic Regression (`C=0.1`, `class_weight='balanced'`)\n"
        "- **Split:** 70 % train / 10 % validation / 20 % test (stratified)\n"
        "- **Imbalance:** SMOTE applied to training set only\n"
        "- **Evaluation:** threshold sweep, confusion matrix, ROC, PR, 5-fold CV\n"
        "- **Explainability:** LR coefficients + optional SHAP\n"
        "- **CV fold scores:** computed live from data (not hardcoded)\n"
    )

# Entry-point

def main():
    st.title("🏥 Hospital Readmission Prediction Dashboard")
    st.caption("DATA 230 - Logistic Regression Track")

    if not os.path.exists(DATA_PATH):
        st.error(
            f"`diabetic_cleaned.csv` not found at `{DATA_PATH}`. "
            "Place the CSV in the same folder as `app.py` and rerun."
        )
        st.stop()

    with st.spinner("Reading CSV → preparing features → training model…"):
        data, X, y, summary_df, _ = load_and_prepare(DATA_PATH)   # STEP 1
        eda_views                  = build_eda_tables(data)         # STEP 2
        outputs                    = train_lr_model(X, y)           # STEP 3

    page = st.sidebar.radio("Navigate", [                           # STEP 4 / 5
        "Overview", "EDA Insights", "Logistic Regression Results",
        "Interpretation", "Methodology Notes",
    ])

    if   page == "Overview":                    render_overview(summary_df, outputs["split_df"])
    elif page == "EDA Insights":                render_eda(eda_views)
    elif page == "Logistic Regression Results": render_results(outputs)
    elif page == "Interpretation":              render_interpretation(outputs)
    else:                                       render_methodology()


if __name__ == "__main__":
    main()
