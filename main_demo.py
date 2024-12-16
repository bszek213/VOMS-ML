import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
import datetime 

SIZE_all = 1.05
plt.rcParams.update({
    'font.size': 14*SIZE_all,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 14*SIZE_all,
    'axes.titleweight': 'bold',
    'legend.fontsize': 14*SIZE_all,
    'legend.title_fontsize': 14*SIZE_all,
    'xtick.labelsize': 12*SIZE_all,
    'ytick.labelsize': 12*SIZE_all,
    'lines.linewidth': 2
})

def read_voms_data(file_path, sheet_name=None):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# def convert_height_to_cm(height):
#     if pd.isna(height):
#         return height
#     if isinstance(height, str):
#         if "'" in height:
#             feet, inches = map(int, re.findall(r'\d+', height))
#             return round(feet * 30.48 + inches * 2.54)
#         elif 'cm' in height:
#             return int(re.findall(r'\d+', height)[0])
#     return height

def convert_height_to_cm(height):
    if pd.isna(height) or height == '':
        return np.nan  # Return NaN if the value is NaN or an empty string
    # Check if the height is a datetime object
    if isinstance(height, datetime.datetime):
        return np.nan  # Return NaN for datetime values
    if isinstance(height, str):
        # Remove everything before the "'" (for heights like "5'7.5")
        height = re.sub(r"^.*?'", "", height)

        # Handle feet and inches format (e.g., "5' 10")
        if "'" in height:
            parts = re.findall(r'\d+', height)
            if len(parts) == 2:  # If both feet and inches are provided
                feet, inches = map(int, parts)
                return round(feet * 30.48 + inches * 2.54)
            elif len(parts) == 1:  # If only inches are provided
                inches = int(parts[0])
                return round(inches * 2.54)  # Convert inches to cm
        # Handle centimeters format (e.g., "175 cm")
        elif 'cm' in height:
            parts = re.findall(r'\d+', height)
            if parts:
                return int(parts[0])  # Return the first number as cm
    # Return the original value if it's not in a recognized format
    return np.nan  # Return NaN if no valid height format is found

file_path_2022_2023 = os.path.join("/media/brianszekely/TOSHIBA EXT/ExperimentData/master_sheets", "2022_23 UNR Master Spreadsheet.xlsx")
file_path_2023_2024 = os.path.join("/media/brianszekely/TOSHIBA EXT/ExperimentData/master_sheets", "2023_24 UNR Master Spreadsheet.xlsx")
file_path_2024_2025 = os.path.join("/media/brianszekely/TOSHIBA EXT/ExperimentData/master_sheets", "2024-25 UNR Master Spreadsheet.xlsx")

symptoms_2022_2023 = read_voms_data(file_path_2022_2023, sheet_name='Demographics')
symptoms_2023_2024 = read_voms_data(file_path_2023_2024, sheet_name='Concussion Demographics')
symptoms_2024_2025 = read_voms_data(file_path_2024_2025, sheet_name='Concussion Demo')

all_data = pd.concat([symptoms_2022_2023, symptoms_2023_2024, symptoms_2024_2025])
all_data.rename(columns={'  ': 'ID'}, inplace=True)
all_data = all_data.loc[:, ~all_data.columns.duplicated()]

src_ids, healthy_ids = pd.read_csv('ID_src.csv'), pd.read_csv('ID_healthy.csv')

# filtered_src = all_data[all_data.apply(lambda row: row.astype(str).str.contains('|'.join(src_ids)).any(), axis=1)]
# filtered_healthy = all_data[all_data.apply(lambda row: row.astype(str).str.contains('|'.join(src_ids)).any(), axis=1)]
filtered_src = all_data[all_data['ID'].astype(str).str[:3].isin(src_ids['ID'].astype(str).str[:3])]
filtered_src = filtered_src[filtered_src['Visit Type'].str.contains('CON',na=False)]

filtered_controls = all_data[all_data['ID'].str[:4].isin(healthy_ids['ID'].str[:4])]
filtered_controls = filtered_controls[filtered_controls['Visit Type'].str.contains('Base',na=False)]

df_control = filtered_controls[['Days Since Injury','Sport', 'Gender', 'Age',
       'Height', 'Kg']]

df_control['Height'] = df_control['Height'].apply(convert_height_to_cm).astype(float)
df_control['Height'] = df_control['Height'].fillna(df_control['Height'].mean())
df_control['Gender'] = df_control['Gender'].str.strip()
df_control['Age'] = df_control['Age'].where((df_control['Age'] <= 100) & (df_control['Age'] > 17), np.nan)
df_control['Age'] = df_control['Age'].fillna(df_control['Age'].mean())
df_src = filtered_src[['Days Since Injury','Sport', 'Gender', 'Age',
       'Height', 'Kg']]

df_src['Height'] = df_src['Height'].apply(convert_height_to_cm).astype(float)
df_src['Height'] = df_src['Height'].fillna(df_control['Height'].mean())
df_src['Days Since Injury'] = df_src['Days Since Injury'].fillna(df_src['Days Since Injury'].median())
df_src['Days Since Injury'] = df_src['Days Since Injury'].where(df_src['Days Since Injury'] <= 14, np.nan)
df_src['Days Since Injury'] = df_src['Days Since Injury'].fillna(df_src['Days Since Injury'].median())
df_src['Gender'] = df_src['Gender'].str.strip()
df_src['Age'] = df_src['Age'].where((df_src['Age'] <= 100) & (df_src['Age'] > 17), np.nan)
df_src['Age'] = df_src['Age'].fillna(df_src['Age'].mean())

#print out 
numeric_cols_control = df_control.select_dtypes(include=['float64', 'int64']).columns
print("Mean and Std for df_control:")
for col in numeric_cols_control:
    print(f"{col} - Mean: {df_control[col].mean()}, Std: {df_control[col].std()}")

# Calculate mean and std for numeric columns in df_src
numeric_cols_src = df_src.select_dtypes(include=['float64', 'int64']).columns
print("\nMean and Std for df_src:")
for col in numeric_cols_src:
    print(f"{col} - Mean: {df_src[col].mean()}, Std: {df_src[col].std()}")

# Calculate value counts as proportions for categorical columns
categorical_cols_control = df_control.select_dtypes(include=['object']).columns
print("\nValue counts as proportions for categorical columns in df_control:")
for col in categorical_cols_control:
    print(f"{col} - Proportions:\n{df_control[col].value_counts(normalize=True)}")

categorical_cols_src = df_src.select_dtypes(include=['object']).columns
print("\nValue counts as proportions for categorical columns in df_src:")
for col in categorical_cols_src:
    print(f"{col} - Proportions:\n{df_src[col].value_counts(normalize=True)}")

plt.figure(figsize=(12, 10))

# Age Distribution for both groups
plt.subplot(2, 2, 1)
sns.histplot(data=df_control, x='Age', kde=True, stat='count', common_norm=False, color='blue', label='Control', alpha=0.5)
sns.histplot(data=df_src, x='Age', kde=True, stat='count', common_norm=False, color='orange', label='SRC', alpha=0.5)
plt.xlabel('Age (years)')
plt.legend()
plt.text(-0.1, 1.1, 'A', transform=plt.gca().transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# Gender Distribution (relative proportions) for both groups
plt.subplot(2, 2, 2)
gender_counts_control = df_control['Gender'].value_counts(normalize=False)
gender_counts_src = df_src['Gender'].value_counts(normalize=False)
gender_counts = pd.DataFrame({'Control': gender_counts_control, 'SRC': gender_counts_src}).fillna(0)
gender_counts.plot(kind='bar', stacked=False, ax=plt.gca())
# plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.xlabel('')
plt.text(-0.1, 1.1, 'B', transform=plt.gca().transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# Sport Distribution (relative proportions) for both groups
plt.subplot(2, 2, 3)
sport_counts_control = df_control['Sport'].value_counts(normalize=False)
sport_counts_src = df_src['Sport'].value_counts(normalize=False)
sport_counts = pd.DataFrame({'Control': sport_counts_control, 'SRC': sport_counts_src}).fillna(0)
sport_counts.plot(kind='bar', stacked=False, ax=plt.gca())
plt.xlabel('Sport')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.text(-0.1, 1.2, 'C', transform=plt.gca().transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# Days Since Injury Distribution for only the src group
plt.subplot(2, 2, 4)
sns.histplot(data=df_src, x='Days Since Injury', kde=True, stat='count', common_norm=False, color='green')
plt.xlabel('Days Since Injury - SRC Group')
plt.text(-0.1, 1.2, 'D', transform=plt.gca().transAxes, fontsize=18, fontweight='bold', va='top', ha='right')

# Finalize the layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join('figures', 'demo_figure_combined.png'), dpi=400)
