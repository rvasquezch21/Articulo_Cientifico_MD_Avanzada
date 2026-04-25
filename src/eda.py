
import pandas as pd

def class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df[target_col].value_counts().reset_index()
    out.columns = [target_col, "count"]
    out["percentage"] = (out["count"] / out["count"].sum()).round(4)
    return out

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").describe().T.round(4)
