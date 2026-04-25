import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


def _bin_numeric(value, low, high, prefix):
    try:
        v = float(value)
    except Exception:
        return f"{prefix}=unknown"

    if v < low:
        return f"{prefix}=low"
    if v <= high:
        return f"{prefix}=medium"
    return f"{prefix}=high"


def build_transactions(df: pd.DataFrame, target_col: str = "NObeyesdad") -> list[list[str]]:
    transactions = []

    categorical_cols = [
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
        "web_mining_risk_profile",
    ]

    for _, row in df.iterrows():
        items = []

        for col in categorical_cols:
            if col in df.columns:
                items.append(f"{col}={row[col]}")

        if "Age" in df.columns:
            items.append(_bin_numeric(row["Age"], 20, 30, "Age"))

        if "Weight" in df.columns:
            items.append(_bin_numeric(row["Weight"], 60, 90, "Weight"))

        if "FAF" in df.columns:
            items.append(_bin_numeric(row["FAF"], 1, 2, "FAF"))

        if "TUE" in df.columns:
            items.append(_bin_numeric(row["TUE"], 1, 2, "TUE"))

        if "CH2O" in df.columns:
            items.append(_bin_numeric(row["CH2O"], 1, 2, "CH2O"))

        if target_col in df.columns:
            items.append(f"{target_col}={row[target_col]}")

        transactions.append(items)

    return transactions


def mine_rules(
    transactions,
    min_support=0.10,
    min_confidence=0.50,
    min_lift=1.10,
    max_len=3,
    target_prefix="NObeyesdad="
):
    te = TransactionEncoder()
    encoded = te.fit(transactions).transform(transactions, sparse=True)

    basket = pd.DataFrame.sparse.from_spmatrix(
        encoded,
        columns=te.columns_
    )

    frequent = fpgrowth(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    if frequent.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return pd.DataFrame()

    rules = rules[rules["lift"] >= min_lift].copy()

    # Nos quedamos principalmente con reglas que predicen clase de obesidad
    rules = rules[
        rules["consequents"].apply(
            lambda x: any(str(item).startswith(target_prefix) for item in x)
        )
    ].copy()

    if rules.empty:
        return pd.DataFrame()

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )
    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )

    keep_cols = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction"
    ]

    rules = rules[keep_cols]
    rules = rules.sort_values(["lift", "confidence"], ascending=False)
    rules = rules.round(4).reset_index(drop=True)

    return rules.head(100)