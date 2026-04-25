
import pandas as pd
import numpy as np

def enrichment_dictionary() -> pd.DataFrame:
    rows = [
        {
            "source_concept": "Diet quality / caloric excess",
            "dataset_variables": "FAVC, FCVC, NCP, CAEC",
            "generated_feature": "dietary_risk_score",
            "logic": "Higher risk when frequent high-calorie food, low vegetable consumption, frequent snacking or irregular meals are observed."
        },
        {
            "source_concept": "Physical inactivity",
            "dataset_variables": "FAF",
            "generated_feature": "physical_inactivity_score",
            "logic": "Higher risk when physical activity frequency is low."
        },
        {
            "source_concept": "Sedentary screen behavior",
            "dataset_variables": "TUE",
            "generated_feature": "screen_time_risk_score",
            "logic": "Higher risk when technology use time is high."
        },
        {
            "source_concept": "Alcohol / tobacco lifestyle risk",
            "dataset_variables": "CALC, SMOKE",
            "generated_feature": "substance_risk_score",
            "logic": "Higher risk when alcohol consumption or smoking behavior is present."
        },
        {
            "source_concept": "Transport-related activity",
            "dataset_variables": "MTRANS",
            "generated_feature": "mobility_risk_score",
            "logic": "Higher risk when transport mode indicates low active mobility."
        },
        {
            "source_concept": "Family predisposition",
            "dataset_variables": "family_history_with_overweight",
            "generated_feature": "hereditary_risk_score",
            "logic": "Higher risk when family history of overweight is present."
        }
    ]
    return pd.DataFrame(rows)

def _map_caeca(value):
    v = str(value).lower()
    if v == "always":
        return 3
    if v == "frequently":
        return 2
    if v == "sometimes":
        return 1
    return 0

def _map_calc(value):
    v = str(value).lower()
    if v == "always":
        return 3
    if v == "frequently":
        return 2
    if v == "sometimes":
        return 1
    return 0

def add_web_mining_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["dietary_risk_score"] = 0.0
    out["dietary_risk_score"] += np.where(out["FAVC"].astype(str).str.lower() == "yes", 2, 0)
    out["dietary_risk_score"] += np.where(out["FCVC"] < 2, 1.5, 0)
    out["dietary_risk_score"] += np.where(out["NCP"] < 2, 1, 0)
    out["dietary_risk_score"] += out["CAEC"].apply(_map_caeca)

    out["physical_inactivity_score"] = np.where(out["FAF"] < 1, 3, np.where(out["FAF"] < 2, 1.5, 0))
    out["screen_time_risk_score"] = np.where(out["TUE"] > 1.5, 2, np.where(out["TUE"] > 0.5, 1, 0))
    out["substance_risk_score"] = out["CALC"].apply(_map_calc) + np.where(out["SMOKE"].astype(str).str.lower() == "yes", 1, 0)

    low_active_transport = ["Automobile", "Motorbike", "Public_Transportation"]
    out["mobility_risk_score"] = np.where(out["MTRANS"].isin(low_active_transport), 1.5, 0)

    out["hereditary_risk_score"] = np.where(
        out["family_history_with_overweight"].astype(str).str.lower() == "yes",
        2,
        0
    )

    score_cols = [
        "dietary_risk_score",
        "physical_inactivity_score",
        "screen_time_risk_score",
        "substance_risk_score",
        "mobility_risk_score",
        "hereditary_risk_score"
    ]
    out["web_mining_enrichment_score"] = out[score_cols].sum(axis=1)

    out["web_mining_risk_profile"] = pd.cut(
        out["web_mining_enrichment_score"],
        bins=[-0.1, 3, 6, 20],
        labels=["Low enriched risk", "Medium enriched risk", "High enriched risk"]
    ).astype(str)

    return out
