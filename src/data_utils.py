
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out

def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum())
    }
