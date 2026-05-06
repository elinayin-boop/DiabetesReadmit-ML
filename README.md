# 🏥 DiabetesReadmit-ML
 
> Predicting 30-day hospital readmission in diabetic patients using Logistic Regression - featuring full data cleaning, EDA, and interactive dashboards built in Plotly, Tableau, Power BI, and Streamlit.

---

# Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Data Cleaning](#-data-cleaning)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Machine Learning](#-machine-learning)
- [Dashboards](#-dashboards)
- [How to Run](#-how-to-run)
- [Requirements](#-requirements)

## Project Overview

Hospital readmission within 30 days is one of the most costly and preventable problems in modern healthcare. In the U.S., preventable readmissions cost an estimated **$26 billion annually**, and hospitals face penalties under Medicare's Hospital Readmissions Reduction Program (HRRRP) for excessive rates.

This project builds an end-to-end machine learning pipeline to **predict whether a diabetic patient will be readmitted within 30days of discharge**, using clinical and administrative data routinely collected during hospital stays.

### Research Question
> *Can we accurately predict 30-day hospital readmission in diabetic patients using routine clinical data - and which patient features drive that risk most strongly?*

### 🎯 Research Question
> *Can we accurately predict 30-day hospital readmission in diabetic patients using routine clinical data - and which patient features drive that risk most strongly?*

## Key Findings
 - Logistic Regression achieved **ROC-AUC ≈ 0.68** with **Recall ≈ 0.37** at threshold 0.35
 - ***Prior inpatient visits* and **number of medications** are the strongest predictors
 - Only **11.2%** of encounters result in early readmission - severe class imbalance requiring SMOTE
 - Middle-aged groups (40-60) have **higher** early readmission rates than the elderly - counter-intuitive and clinically meaningful
 - LR was chosen for its **interpretability** - critical in healthcare where clinicians need to understand why a patient is flagged.

## 📊 Dataset
 
| Property | Detail |
|---|---|
| **Name** | Diabetes 130-US Hospitals (1999–2008) |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) |
| **Records** | 101,766 patient encounters |
| **Features** | 50 (demographic, clinical, medication, diagnostic) |
| **Target** | `readmitted` → binarized: `<30 days` = 1, else = 0 |
| **Class Balance** | 11.2% positive - severely imbalanced |
| **Time Period** | 1999 - 2008 |
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
│   ├── diabetic_data.csv                            # Raw dataset (download from UCI)
│   ├── IDS_mapping.csv                              # ID mapping file
│   └── diabetic_cleaned.csv                         # Output of cleaning script
│
├── diabetes_hospitsl_readmission_cleaning.py
├── diabetes_plotly_visualizations.py
├── streamlit-app-logistic-regressions.py
│
├── images/
│   ├── ReadmissionRateByAgeGroup(plotlyV1).png
│   ├── MedicationsByReadmissionStatus(plotlyV2).png
│   ├── ReadmissionOutcomeByInsulinUsage(plotlyV3).png
│   ├── ReadmissionRateByHospitalStayLengthAndDiagnosisCount(TableauV4).png
│   ├── Top10MedicalSpecialtiesbyEarlyReadmissionRate(TableauV5).png
│   ├── ReadmissionOutcomesbyPatientRace(TableauV6).png
│   ├── ReadmissionRateByAdmissionSource&Type(PowerBIV7).png
│   ├── LabProcedureIntensityVsReadmissionRate(PowerBIV8).png
│   ├── ClinicalComplexity:HospitalStayVsDiagnosisCount(PowerBIV9).png
│   ├── tableau_dashboard.png
│   └── PowerBI_dashboard.png
│
├── requirements.txt
└── README.md  

```

---

## Pipeline Overview
 
```
Raw Data (diabetic_data.csv + IDS_mapping.csv)
        │
        ▼
[diabetes_hospital_readmission_cleaning.py]
  • Replace ? with NaN
  • Drop weight, encounter_id, patient_nbr
  • Remove deceased patients
  • Impute missing categoricals
  • Binarize target variable
  • Map age to numeric midpoints
        │
        ▼
diabetic_cleaned.csv
        │
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
[diabetes_plotly_               [Tableau Public]              [Power BI Desktop]
 visualizations.py]              Visuals 4, 5, 6               Visuals 7, 8, 9
 Visuals 1, 2, 3
        │
        ▼
[streamlit-app-logistic-regressions.py]
  Trains LR model live
  11 interactive ML visuals across 5 pages
```
 
---

## Data Cleaning

Handled in `diabetes_hospital_readmission_cleaning.py`. Files required `diabetic_data.csv` + `IDS_mapping.csv`. Outputs `diabetic_cleaned.csv`.
 
| Step | Action | Reason |
|---|---|---|
| 1 | Replace `?` with `NaN` | Missing values encoded as `?` not standard NaN |
| 2 | Drop `weight` | 96.9% missing — unusable |
| 3 | Drop `encounter_id`, `patient_nbr` | ID columns, no predictive value |
| 4 | Remove discharge = 11 | Deceased patients — readmission meaningless |
| 5 | Remove `Unknown/Invalid` gender | 3 records, introduces noise |
| 6 | Fill `race`, `medical_specialty`, `payer_code` → `"Unknown"` | Preserve records |
| 7 | Fill `diag_1/2/3` → `"0"` | Indicate missing diagnosis code |
| 8 | Binarize target: `<30` → 1, else → 0 | Binary classification framing |
| 9 | Map age brackets → numeric midpoints | Enable as continuous feature |
| 10 | Convert numeric columns | Correct dtypes for modelling |
 
**Before:** 101,766 rows × 50 columns → **After:** ~100,000 rows × 49 columns

---

## 📈 Exploratory Data Analysis
 
### Plotly Visuals ( `diabetes_plotly_visualizations.py`)
 
**Visual 1 - Readmission Rate by Age Group**
 
![V1](images/ReadmissionRateByAgeGroup(PlotlyV1).png)

Middle-aged groups (40 - 60) show higher early readmission rates than the elderly - likely because older patients receive more intensive post-discharge coordination. Age is an important ML feature.
 
---

**Visual 2 - Medications by Readmission Status**
 
![V2](images/MedicationsByReadmissionStatus(PlotlyV2).png)
 
Patients readmitted within 30 days carry a higher median medication count (~18 vs ~15), reflecting greater disease complexity. `num_medications` is a strong LR predictor.
 
---

**Visual 3 - Readmission Outcome by Insulin Usage**
 
![V3](images/ReadmissionOutcomeByInsulinUsage(PlotlyV3).png)
 
Patients with upward insulin dose adjustments show a higher proportion of early readmissions - signalling worsening glycaemic control during the encounter.
 
---

### Tableau Visuals

Tableau Public Link - https://public.tableau.com/views/DiabetesReadmitVisuals/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link 
 
**Visual 4 - Readmission Rate by Hospital Stay Length and Diagnosis Count**
 
![V4](images/ReadmissionRateByHospitalStayLengthAndDiagnosisCount(TableauV4).png)
 
High stay length combined with high diagnosis count produces the darkest readmission risk cells - combined clinical complexity is a stronger predictor than either feature alone.
 
---
 
**Visual 5 - Top 10 Medical Specialties by Early Readmission Rate**
 
![V5](images/Top10MedicalSpecialtiesByEarlyReadmissionRate(TableauV5).png)
 
Certain specialties show systematically higher early readmission rates, suggesting they treat higher-acuity patients or have gaps in discharge planning protocols.
 
---
 
**Visual 6 - Readmission Outcomes by Patient Race**
 
![V6](images/ReadmissionOutcomesByPatientRace(TableauV6).png)
 
Demographic disparities in early readmission proportions are visible across racial groups — an equity consideration for any clinical deployment of the model.
 
**Full Tableau Dashboard:**
 
![Tableau Dashboard](images/tableau_dashboard.png)
 
---

### Power BI Visuals
 
**Visual 7 - Readmission Rate by Admission Source and Type**
 
![V7](images/ReadmissionRateByAdmissionSource&Type(PowerBIV7).png)
 
Emergency admissions show the highest early readmission rates - admission context is a meaningful predictor.
 
---
 
**Visual 8 - Lab Procedure Intensity vs Readmission Rate**
 
![V8](images/LabProcedureIntensityVsReadmissionRate(PowerBIV8).png)
 
Higher lab procedure counts correlate with increased readmission risk - reflecting greater diagnostic workup for more severely ill patients.
 
---
 
**Visual 9 - Clinical Complexity: Hospital Stay vs Diagnosis Count**
 
![V9](images/ClinicalComplexity:HospitalStayVsDiagnosisCount(PowerBIV9).png)
 
The combined complexity marker of stay length and diagnosis count predicts readmission risk - consistent with the Tableau heatmap finding.
 
**Full Power BI Dashboard:**
 
![Power BI Dashboard](images/PowerBI_dashboard.png)
 
---

## Machine Learning
 
All ML logic is inside `streamlit-app-logistic-regressions.py`. It trains directly from `diabetic_cleaned.csv` every time the app loads, cached with `@st.cache_data`.

### Why Logistic Regression?
 
LR coefficients directly explain which features increase or decrease readmission probability - in healthcare, clinicians need to understand *why* a patient is flagged, not just *that* they are. LR provides that transparency without sacrificing meaningful performance.

### Modeling Approach
 
```
diabetic_cleaned.csv
        │
        ▼
Stratified split: 70% train | 10% validation | 20% test
        │
        ▼
SMOTE on training set ONLY → balanced 50/50 classes (no leakage)
        │
        ▼
StandardScaler fitted on train only → applied to val and test
        │
        ▼
Logistic Regression (C=0.1, class_weight='balanced', max_iter=1000)
        │
        ├── Threshold sweep on validation: 0.10 / 0.25 / 0.35 / 0.50
        │   Chosen: 0.35 (maximises recall, improves precision)
        │
        └── Final evaluation on untouched test set
            + 5-fold stratified cross-validation for stability
```
 
### Results (threshold = 0.35)
 
| Metric | Score |
|---|---|
| ROC-AUC | ~0.68 |
| Recall | ~0.37 |
| F1 (positive) | ~0.26 |
| Precision | ~0.20 |
| Balanced Accuracy | ~0.65 |
 
*Exact values computed live from your data when the Streamlit app runs.*
 
---
 
## Dashboards
 
| Dashboard | Tool | Visuals | Type |
|---|---|---|---|
| EDA | **Plotly** (Colab) | V1, V2, V3 | EDA |
| EDA | **Tableau** | V4, V5, V6 | EDA |
| ML Results | **Power BI** | V7, V8, V9 | ML |
| ML Results | **Streamlit** *(self-learned)* | 11 live visuals | ML |

 ### Run Streamlit Locally
 
```bash
pip install -r requirements.txt
streamlit run streamlit-app-logistic-regressions.py
# Opens at http://localhost:8501
```

### Run Streamlit on Google Colab
 
```python
# Cell 1
!pip install -q streamlit imbalanced-learn shap plotly pyngrok
 
# Cell 2 — upload app + diabetic_cleaned.csv to /content/ first
from pyngrok import ngrok
import subprocess, threading, time
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")
threading.Thread(
    target=lambda: subprocess.run(
        ["streamlit", "run", "/content/streamlit-app-logistic-regressions.py",
         "--server.port", "8501", "--server.headless", "true"]
    ), daemon=True
).start()
time.sleep(4)
print("Open →", ngrok.connect(8501))
```
 
---

## Requirements
 
```
pandas
numpy
scikit-learn
imbalanced-learn
plotly
streamlit
shap
pyngrok
```
 
```bash
pip install -r requirements.txt
```
 
---

