import pandas as pd

# Load CSV
df = pd.read_csv(r"C:\Users\HP\OneDrive\文档\TY AI&DS\Project 1\PMF P2 App\Economic Dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert wide → long
df_long = df.melt(
    id_vars=["Country Name", "Country Code", "Series Name"],
    var_name="Year",
    value_name="Value"
)

# Extract year
df_long["Year"] = df_long["Year"].str.extract('(\d+)')

# 🔥 MOST IMPORTANT FIX
df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")

# Drop missing values
df_long = df_long.dropna(subset=["Value"])

# Pivot
df_final = df_long.pivot_table(
    index=["Country Name", "Year"],
    columns="Series Name",
    values="Value"
).reset_index()

print(df_final.head())

# Save
df_final.to_csv("clean_economic_data.csv", index=False)