## Overview

Hospital readmission within 30 days is one of the most costly and preventable problems in modern healthcare. In the U.S. alone, preventable readmissions cost an estimated **$26 billion annually**, and hospitals face penalties under Medicare's Hospital Readmissions Reduction Program (HRRP) for excessive rates.

This project builds an end-to-end machine learning pipeline to **predict whether a diabetic patient will be readmitted within 30 days of discharge**, using clinical and administrative data routinely collected during hospital stays.

### 🎯 Research Question
> *Can we accurately predict 30-day hospital readmission in diabetic patients using routine clinical data - and which patient features drive that risk most strongly?*

## 📊 Dataset

| Property | Detail |
|---|---|
| **Name** | Diabetes 130-US Hospitals (1999–2008) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) |
| **Records** | 101,766 patient encounters |
| **Features** | 50 (demographic, clinical, medication, diagnostic) |
| **Target** | `readmitted` → binarized to `<30 days` vs. `not` |
| **Class Balance** | 11.2% positive (readmitted <30d) - severely imbalanced |
| **Time Period** | 1999 – 2008 |
| **Hospitals** | 130 U.S. hospitals and integrated delivery networks |

### Target Variable Distribution

| Class | Count | Percentage |
|---|---|---|
| NO (not readmitted) | 54,864 | 53.9% |
| >30 days | 35,545 | 34.9% |
| <30 days *(positive class)* | 11,357 | 11.2% |

---

## 📁 Project Structure

```
DiabetesReadmit-ML/
│
├── data/
│   ├── diabetic_data.csv              # Raw dataset 
│   ├── IDS_mapping.csv                # ID mapping file
│   └── diabetic_cleaned.csv           # Cleaned dataset 
│
├── images/
│   ├── ReadmissionOutcomesbyPatientRace(TablueauV6).png
│   ├── ReadmissionRateByHospitalStayLengthAndDiagnosisCount(TablueauV4).png
│   ├── Top10MedicalSpecialtiesByEarlyReadmissionRate(TableauV5).png
│   ├── tablueay_dashboard.png
│
├── diabetes_hospital_readmission_cleaning.py
│
├──  diabetes_plotly_visualizations.py
│
└── README.md
```

---

## Data Cleaning

All cleaning is handled in `diabetes_hospital_readmission_cleaning.py`. Run it first - it outputs `diabetic_cleaned.csv`.

| Step | Action | Reason |
|---|---|---|
| 1 | Replace `?` with `NaN` | Missing values were encoded as `?` |
| 2 | Drop `weight` column | 96.9% missing - unusable |
| 3 | Drop `encounter_id`, `patient_nbr` | ID columns, no predictive value |
| 4 | Remove deceased patients (discharge = 11, 13, 14) | Readmission irrelevant for deceased |
| 5 | Remove `Unknown/Invalid` gender | Only 3 records, noise |
| 6 | Fill `race`, `medical_specialty`, `payer_code` → `"Unknown"` | Preserve records, signal absence |
| 7 | Fill `diag_1/2/3` → `"0"` | Indicate missing diagnosis code |
| 8 | Binarize target: `<30` → 1, else → 0 | Frame as binary classification |
| 9 | Map age brackets → numeric midpoints | Enable use as continuous feature |

**Before cleaning:** 101,766 rows × 50 columns
**After cleaning:** ~100,121 rows × 49 columns

---

## 📈 Exploratory Data Analysis

Three Plotly visualizations were created in `diabetes_plotly_visualizations.py` and three additional Tableau visualizations in `diabetes_hospital_readmission_cleaning.py`, completing the EDA dashboard.

### Plotly Visualizations 

| # | Chart | Key Insight |
|---|---|---|
| 1 | Bar: Readmission rate by age group | Middle-aged groups (40–60) have *higher* early readmission than elderly - counter-intuitive |

| 2 | Box plot: Medications by readmission status | Patients readmitted <30d have ~18 median medications vs ~15 for non-readmitted |
| 3 | Grouped bar: Insulin usage vs readmission | "Up" insulin adjustments correlate with higher early readmission - signals worsening control |

### Tableau Visualizations

| # | Chart | Key Insight |
|---|---|---|
| 4 | Heatmap: Time in hospital × Number of diagnoses | High stay length + high diagnosis count = highest readmission risk cells |

!(images/ReadmissionRateByHospitalStayLengthAndDiagnosisCount(TablueauV4).png)

| 5 | Bar: Top 10 medical specialties by readmission rate | Certain specialties (e.g., Internal Medicine) show systematically higher readmission rates |

!(images/Top10MedicalSpecialtiesByEarlyReadmissionRate(TableauV5).png)

| 6 | 100% Stacked bar: Readmission by race | Demographic disparities visible in <30d readmission proportions across racial groups |

!(images/ReadmissionOutcomesbyPatientRace(TablueauV6).png)

### How EDA Informed Modeling
- Severe class imbalance → **SMOTE required**
- Non-linear age/readmission pattern → **tree-based model preferred over linear**
- Prior utilization gradient → **engineer `prior_utilization` composite feature**

---

## 🤖 Machine Learning

All ML code is in `3_ml_pipeline.py`. Run in Google Colab after `1_data_cleaning.py`.

### Features Used (18 total)

| Type | Features |
|---|---|
| Numeric | `age_numeric`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses`, `prior_utilization`*, `med_change_count`* |
| Categorical | `race`, `gender`, `insulin`, `change`, `diabetesMed`, `diag1_category`*, `A1Cresult`, `max_glu_serum`, `admission_type_id`, `discharge_disposition_id` |

*\* Engineered features*

### Engineered Features

```python
# 1. Prior healthcare burden (proxy for chronic illness severity)
prior_utilization = number_outpatient + number_emergency + number_inpatient

# 2. Medication adjustment activity
med_change_count = count of medication columns where value is 'Up' or 'Down'

# 3. ICD-9 diagnosis category (reduces ~900 codes to 8 clinical groups)
diag1_category = map(diag_1 → Diabetes / Circulatory / Respiratory / ...)
```

### Modeling Approach

```
Cleaned Data
     │
     ▼
Stratified Train/Test Split (80% / 20%)
     │
     ├── Training set → SMOTE → Balanced training set (50/50)
     │                              │
     │              ┌───────────────┴────────────────┐
     │              ▼                                ▼
     │     Logistic Regression              Random Forest
     │     (StandardScaler +                (200 trees,
     │      L2, C=1.0)                       max_depth=15,
     │                                       class_weight='balanced')
     │
     └── Test set (untouched, true distribution)
              │
              ▼
         Evaluation
```

> ⚠️ **Important:** SMOTE is applied **after** the train/test split to prevent data leakage into the test set.

### Model Results

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **ROC-AUC** | 0.6821 | **0.7143** |
| **F1 Score** | 0.2634 | **0.2891** |
| **Recall** | 0.3712 | **0.4105** |
| **Precision** | 0.2041 | **0.2218** |

### Top 8 Features by Importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `prior_utilization` | 0.1423 |
| 2 | `num_medications` | 0.1187 |
| 3 | `time_in_hospital` | 0.1052 |
| 4 | `num_lab_procedures` | 0.0934 |
| 5 | `age_numeric` | 0.0871 |
| 6 | `number_diagnoses` | 0.0812 |
| 7 | `med_change_count` | 0.0754 |
| 8 | `num_procedures` | 0.0698 |

---

## 📊 Dashboards

Four dashboards were built across four different tools, satisfying the requirement for 3 covered-in-class tools + 1 self-learned tool.

| Dashboard | Tool | Visuals | Audience |
|---|---|---|---|
| Midterm 1 | **Plotly** (Google Colab) | Class imbalance, medications box plot, insulin bar | EDA stage |
| Midterm 2 | **Tableau** | Heatmap, specialty bar, race stacked bar | EDA stage |
| Final 1 | **Power BI** | Model metrics comparison, feature importance, risk donut + KPIs | ML results |
| Final 2 | **Streamlit** *(self-learned)* | ROC curves, risk score histogram, actual vs predicted by group | ML results |

### Running the Streamlit Dashboard

```bash
# 1. Install dependencies
pip install streamlit plotly pandas

# 2. Place these 4 files in the same folder as streamlit_app.py:
#    ml_results.csv, feature_importances.csv, model_metrics.csv, roc_curves.csv

# 3. Run
streamlit run 5_streamlit_app.py

# 4. Opens at http://localhost:8501
```

### Streamlit Dashboard Features
- 🔴 **ROC Curve** - both models overlaid with AUC scores
- 📊 **Risk Score Histogram** - predicted probabilities split by true outcome
- 📉 **Actual vs Predicted** - grouped bar by age / diagnosis / risk tier / gender (dropdown)
- 🎛️ **Sidebar filters** - risk tier, gender, diagnosis category (updates all charts live)
- 📌 **KPI cards** - total patients, model accuracy, % high risk, avg risk score

---

## ▶️ How to Run

### Step 1 - Clean the Data (Google Colab)
```python
# In Google Colab:
# 1. Upload diabetic_data.csv when prompted
# 2. Run all cells in 1_data_cleaning.py
# 3. Download diabetic_cleaned.csv
```

### Step 2 - EDA Visualizations (Google Colab)
```python
# 1. Upload diabetic_cleaned.csv when prompted
# 2. Run all cells in 2_plotly_visualizations.py
# 3. Charts render inline in Colab
```

### Step 3 - Train ML Models (Google Colab)
```python
# 1. Upload diabetic_cleaned.csv when prompted
# 2. Run all cells in 3_ml_pipeline.py
# 3. Cell 14 auto-downloads 4 CSV files:
#    ml_results.csv, feature_importances.csv,
#    model_metrics.csv, roc_curves.csv
```

### Step 4 - Streamlit Dashboard (Local)
```bash
pip install -r requirements.txt
streamlit run dashboard/5_streamlit_app.py
```

### Step 5 - Power BI Dashboard
Load the 3 output CSVs into Power BI Desktop:
- `ml_results.csv` → risk distribution donut + KPI cards
- `feature_importances.csv` → feature importance bar chart
- `model_metrics.csv` → model comparison bar chart

### Step 6 - Tableau Dashboard
Load `diabetic_cleaned.csv` into Tableau Public for the 3 EDA visualizations.

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn
imbalanced-learn
plotly
streamlit
```

Install all at once:
```bash
pip install -r requirements.txt
```

For Google Colab, `imbalanced-learn` is installed inline:
```python
!pip install imbalanced-learn --quiet
```



<p align="center">
  Built with Python · Scikit-learn · Plotly · Streamlit · Tableau · Power BI
</p>
