import pandas as pd
from openpyxl import load_workbook

# ==== File paths ====
backtest_file = "backtest_df.xlsx"
features_file = "feature_backtest.csv"
output_file = "backtest_df_updated.xlsx"   # save as Excel (not CSV)

# ==== Read ====
df_backtest = pd.read_excel(backtest_file, sheet_name="Sheet1")
df_features = pd.read_csv(features_file)

# ---- Rename matching key columns ----
df_backtest.rename(columns={df_backtest.columns[1]: "date",
                            df_backtest.columns[2]: "symbol"}, inplace=True)
df_features.rename(columns={df_features.columns[7]: "date",
                            df_features.columns[6]: "symbol"}, inplace=True)

# ---- Convert dates ----
df_backtest["date"] = pd.to_datetime(df_backtest["date"], errors="coerce").dt.normalize()
df_features["date"] = pd.to_datetime(df_features["date"], errors="coerce", dayfirst=True).dt.normalize()

# ---- Get columns I to BV (8 to 73 in 0-based indexing) ----
cols_to_copy = df_features.columns[8:85]

# ---- Merge ----
df_merged = pd.merge(
    df_backtest,
    df_features[["date", "symbol"] + list(cols_to_copy)],
    on=["date", "symbol"],
    how="left"
)

# ---- Place features from col N ----
N_index = 13
df_final = pd.concat([df_merged.iloc[:, :N_index], df_merged[cols_to_copy]], axis=1)

# ==== Save to Excel ====
df_final.to_excel(output_file, index=False)

# ==== Insert formulas in column L (12th column) only where action == BUY ====
wb = load_workbook(output_file)
ws = wb.active

# Find column indexes
col_action = 4   # D column
col_symbol = 3   # C column
col_pnl_pct = 12 # L column

max_row = ws.max_row

for row in range(2, max_row + 1):  # skip header
    action_value = ws.cell(row=row, column=col_action).value
    if action_value == "BUY":
        symbol_cell = f"C{row}"
        lookup_range = f"C{row+1}:L{min(max_row, row+1499)}"
        formula = f'=VLOOKUP({symbol_cell},{lookup_range},10,FALSE)'
        ws.cell(row=row, column=col_pnl_pct).value = formula
    # else leave existing value as is

wb.save(output_file)

print(f"✅ Updated Excel with formulas saved to {output_file}")
