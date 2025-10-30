from pathlib import Path
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "cars_better_dataset.xlsx"
MODEL = BASE / "model_pipeline.joblib"

def load_assets():
    df = pd.read_excel(DATA)
    pipe = joblib.load(MODEL)
    return df, pipe

def recommend_by_budget(budget_usd: float, top_n: int = 30, filters: dict | None = None):
    df, pipe = load_assets()
    mask = pd.Series(True, index=df.index)
    if filters:
        for k, v in filters.items():
            if v is None:
                continue
            if k == "year_min":
                mask &= df["year"] >= v
            elif k == "year_max":
                mask &= df["year"] <= v
            elif k == "mileage_max":
                mask &= df["mileage_km"] <= v
            else:
                if k in df.columns:
                    mask &= df[k] == v
    df_f = df[mask].copy()
    if df_f.empty:
        return df_f
    X = df_f.drop(columns=["price_usd"])
    df_f["pred_price_usd"] = pipe.predict(X)
    affordable = df_f[df_f["pred_price_usd"] <= budget_usd].copy()
    affordable.sort_values(by=["pred_price_usd", "year"], ascending=[True, False], inplace=True)
    return affordable.head(top_n)

def predict_price_for_input(sample: dict):
    _, pipe = load_assets()
    X = pd.DataFrame([sample])
    return float(pipe.predict(X)[0])
