import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "cars_better_dataset.xlsx"

def load_data():
    return pd.read_excel(DATA)

def build_pipeline(df):
    y = df["price_usd"].values
    X = df.drop(columns=["price_usd"])
    num_cols = ["year", "mileage_km", "engine_cc"]
    cat_cols = ["brand", "model", "condition", "body_type", "transmission", "fuel", "city"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                       solver="adam", random_state=42, max_iter=800)

    return Pipeline([("pre", pre), ("mlp", mlp)]), X, y

def train():
    df = load_data()
    pipe, X, y = build_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"MAE (USD): {mae:.2f}")
    print(f"R2: {r2:.3f}")
    joblib.dump(pipe, BASE / "model_pipeline.joblib")
    print("✅ Modelo guardado en model_pipeline.joblib")

if __name__ == "__main__":
    train()
