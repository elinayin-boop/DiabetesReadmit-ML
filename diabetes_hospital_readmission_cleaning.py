import pandas as pd
import numpy as np

# Load Data

df = pd.read_csv('diabetic_data.csv')
ids_map = pd.read_csv('IDS_mapping.csv')

print("Original Shape:", df.shape)
print(df.head(3))

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

print("\nMissing values per column (before cleaning):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Drop Columns
# 'weight' is missing in ~97% of rows — not usable
# 'encounter_id' and 'patient_nbr' are IDs, not features
df.drop(columns=['weight', 'encounter_id', 'patient_nbr'], inplace=True)

print("Dropped: weight, encounter_id, patient_nbr")

# Fill Missing Values
df['race'].fillna('Unknown', inplace=True)
df['medical_specialty'].fillna('Unknown', inplace=True)
df['payer_code'].fillna('Unknown', inplace=True)
df['diag_1'].fillna('0', inplace=True)
df['diag_2'].fillna('0', inplace=True)
df['diag_3'].fillna('0', inplace=True)

print("Filled missing: race, medical_specialty, payer_code, diag_1/2/3")

# Remove Invalid Rows
# Remove unknown/invalid gender entries
df = df[df['gender'] != 'Unknown/Invalid']

# Remove patients who died (discharge_disposition_id == 11)
# Readmission is meaningless for deceased patients
df = df[df['discharge_disposition_id'] != 11]

print("Removed: invalid gender rows, deceased patients (discharge=11)")

# Encode Target Variable
# Binary: readmitted within 30 days = 1, otherwise = 0
df['readmitted_binary'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

print("\nReadmission distribution:")
print(df['readmitted'].value_counts())
print("\nBinary target distribution:")
print(df['readmitted_binary'].value_counts())

# Map Age to Numeric
age_map = {
    '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_map)

print("Added age_numeric column")

# Fix Column Types
num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

print("Converted numeric columns")

# Final Check & Save
print("\nCleaned Shape:", df.shape)
print("Remaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df.to_csv('diabetic_cleaned.csv', index=False)
print("\n Saved cleaned dataset!")
