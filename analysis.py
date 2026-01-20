import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os

# --- STEP 1: LOAD DATA ---
print("🚀 Starting Analysis... Searching for files...")

# Find all CSVs recursively
enrol_files = glob.glob(os.path.join("data", "**", "*enrolment*.csv"), recursive=True)
demo_files = glob.glob(os.path.join("data", "**", "*demographic*.csv"), recursive=True)
bio_files = glob.glob(os.path.join("data", "**", "*biometric*.csv"), recursive=True)

print(f"   Found {len(enrol_files)} Enrolment files.")
print(f"   Found {len(demo_files)} Demographic files.")
print(f"   Found {len(bio_files)} Biometric files.")

def load_merge(file_list):
    if not file_list: return pd.DataFrame()
    # Using 'on_bad_lines' to skip any messy rows
    return pd.concat([pd.read_csv(f, on_bad_lines='skip') for f in file_list], ignore_index=True)

print("⏳ Reading CSV files...")
df_enrol = load_merge(enrol_files)
df_demo = load_merge(demo_files)
df_bio = load_merge(bio_files)

# --- FIX: STANDARDIZE COLUMN NAMES ---
print("🔧 Fixing Column Names...")
for df in [df_enrol, df_demo, df_bio]:
    # 1. Remove spaces
    df.columns = df.columns.str.strip()
    # 2. Convert to lowercase (so 'State' and 'state' become the same)
    df.columns = df.columns.str.lower()
    # 3. Rename specific columns to match our script
    df.rename(columns={'state': 'State', 'district': 'District'}, inplace=True)

# Debug: Check if it worked
print(f"   Enrolment Columns: {list(df_enrol.columns[:5])}")

# --- STEP 2: AGGREGATE & MERGE ---
print("⚙️  Processing Metrics...")

# Group by State & District
enrol_agg = df_enrol.groupby(['State', 'District']).sum(numeric_only=True).reset_index()
demo_agg = df_demo.groupby(['State', 'District']).sum(numeric_only=True).reset_index()
bio_agg = df_bio.groupby(['State', 'District']).sum(numeric_only=True).reset_index()

# Calculate Totals (Summing age columns dynamically)
# We assume the first 2 columns are State/District, the rest are counts
enrol_agg['Total_Enrolment'] = enrol_agg.iloc[:, 2:].sum(axis=1)
demo_agg['Total_Demo_Updates'] = demo_agg.iloc[:, 2:].sum(axis=1)
bio_agg['Total_Bio_Updates'] = bio_agg.iloc[:, 2:].sum(axis=1)

# Merge into Master DataFrame
master_df = pd.merge(enrol_agg[['State', 'District', 'Total_Enrolment']], 
                     demo_agg[['State', 'District', 'Total_Demo_Updates']], on=['State', 'District'], how='left')
master_df = pd.merge(master_df, bio_agg[['State', 'District', 'Total_Bio_Updates']], on=['State', 'District'], how='left')
master_df.fillna(0, inplace=True)

# --- STEP 3: CALCULATE METRICS ---
master_df['Total_Updates'] = master_df['Total_Demo_Updates'] + master_df['Total_Bio_Updates']
master_df = master_df[master_df['Total_Enrolment'] > 0] # Remove zeros
master_df['Update_Intensity'] = (master_df['Total_Updates'] / master_df['Total_Enrolment']) * 1000

# Anomaly Detection (Z-Score)
master_df['Z_Score'] = (master_df['Update_Intensity'] - master_df['Update_Intensity'].mean()) / master_df['Update_Intensity'].std()

print("\n TOP 5 ANOMALIES (Copy this for your Report):")
print(master_df[['State', 'District', 'Update_Intensity', 'Z_Score']].sort_values(by='Z_Score', ascending=False).head(5))

# --- STEP 4: SAVE CHARTS ---
sns.set_style("whitegrid")

# Chart 1: Anomalies
plt.figure(figsize=(10, 6))
sns.histplot(master_df['Update_Intensity'], bins=50, kde=True, color='teal')
plt.axvline(master_df['Update_Intensity'].mean() + 3*master_df['Update_Intensity'].std(), color='red', linestyle='--')
plt.title("Update Intensity Distribution (Red Line = High Stress Zones)")
plt.xlabel("Updates per 1000 Enrolments")
plt.savefig("Chart1_Anomalies.png")

# Chart 2: Prediction
plt.figure(figsize=(10, 6))
sns.regplot(x=master_df['Total_Enrolment'], y=master_df['Total_Bio_Updates'], color='purple', scatter_kws={'alpha':0.5})
plt.title("Predictive Model: Enrolment vs Future Biometric Demand")
plt.xlabel("Total Enrolment")
plt.ylabel("Biometric Updates")
plt.savefig("Chart2_Prediction.png")

print("\n✅ DONE. Charts saved to folder.")