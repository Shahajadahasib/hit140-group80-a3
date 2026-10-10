# Convert the two Objective 2 datasets from Excel to CSV
# (Yakub, Objective 1 Padlet: datasets are saved as .csv).
# The .xlsx files are kept; the main script reads the .csv copies.
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
EXPECTED = {"FIFA_WC2026_Dataset_2_1_104rows": 104,
            "FIFA_WC2026_Dataset_2_2_208rows": 208}

for name, rows in EXPECTED.items():
    df = pd.read_excel(os.path.join(DATA, name + ".xlsx"))
    assert len(df) == rows, f"{name}: expected {rows} rows, got {len(df)}"
    assert df.isnull().sum().sum() == 0, f"{name}: has missing values"
    df.to_csv(os.path.join(DATA, name + ".csv"), index=False)
    print(name, df.shape, "saved as CSV")
