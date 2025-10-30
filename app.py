import streamlit as st
import pandas as pd
from recommender import recommend_by_budget, predict_price_for_input, load_assets

st.set_page_config(page_title="Recomendador de Autos — Red Neuronal", layout="wide")
st.title("🚗 Recomendador de Autos por Presupuesto — MLP Neural Network")

st.markdown("**Ingresa tu presupuesto en USD** y filtra características del auto. "
            "El modelo estima precios reales de autos en el mercado peruano y "
            "te muestra los que puedes comprar según tu dinero disponible.")

with st.sidebar:
    st.header("💰 Tu presupuesto")
    budget = st.number_input("Presupuesto (USD)", min_value=1000, max_value=100000, value=15000, step=500)

    st.header("⚙️ Filtros opcionales")
    df, _ = load_assets()
    brand = st.selectbox("Marca", ["(Cualquiera)"] + sorted(df["brand"].unique().tolist()))
    city = st.selectbox("Ciudad", ["(Cualquiera)"] + sorted(df["city"].unique().tolist()))
    condition = st.selectbox("Condición", ["(Cualquiera)"] + sorted(df["condition"].unique().tolist()))
    body = st.selectbox("Carrocería", ["(Cualquiera)"] + sorted(df["body_type"].unique().tolist()))
    fuel = st.selectbox("Combustible", ["(Cualquiera)"] + sorted(df["fuel"].unique().tolist()))
    trans = st.selectbox("Transmisión", ["(Cualquiera)"] + sorted(df["transmission"].unique().tolist()))
    year_min = st.number_input("Año mínimo", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=2012)
    year_max = st.number_input("Año máximo", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=2025)
    mileage_max = st.number_input("Kilometraje máximo (km)", min_value=0, max_value=int(df["mileage_km"].max()), value=120000, step=5000)

    filters = {
        "brand": None if brand == "(Cualquiera)" else brand,
        "city": None if city == "(Cualquiera)" else city,
        "condition": None if condition == "(Cualquiera)" else condition,
        "body_type": None if body == "(Cualquiera)" else body,
        "fuel": None if fuel == "(Cualquiera)" else fuel,
        "transmission": None if trans == "(Cualquiera)" else trans,
        "year_min": year_min,
        "year_max": year_max,
        "mileage_max": mileage_max,
    }

st.subheader("🔍 Autos dentro de tu presupuesto")

if st.button("Buscar autos"):
    recs = recommend_by_budget(budget, top_n=30, filters=filters)
    if recs.empty:
        st.warning("No se encontraron autos con esos filtros.")
    else:
        st.dataframe(recs[["brand", "model", "year", "mileage_km", "condition", "body_type",
                           "transmission", "fuel", "engine_cc", "city", "pred_price_usd", "price_usd"]])

st.markdown("---")
st.subheader("🧮 Estimar precio de un auto específico")

with st.form("form_predict"):
    brand_i = st.text_input("Marca", "Toyota")
    model_i = st.text_input("Modelo", "Yaris")
    year_i = st.number_input("Año", 2008, 2025, 2018)
    mileage_i = st.number_input("Kilometraje (km)", 0, 300000, 60000)
    condition_i = st.selectbox("Condición", ["Usado", "Nuevo"])
    body_i = st.selectbox("Carrocería", sorted(df["body_type"].unique().tolist()))
    trans_i = st.selectbox("Transmisión", sorted(df["transmission"].unique().tolist()))
    fuel_i = st.selectbox("Combustible", sorted(df["fuel"].unique().tolist()))
    engine_i = st.number_input("Cilindrada (cc)", 800, 5000, 1600)
    city_i = st.selectbox("Ciudad", sorted(df["city"].unique().tolist()))
    submitted = st.form_submit_button("Estimar precio")

    if submitted:
        sample = {
            "brand": brand_i, "model": model_i, "year": int(year_i), "mileage_km": int(mileage_i),
            "condition": condition_i, "body_type": body_i, "transmission": trans_i,
            "fuel": fuel_i, "engine_cc": int(engine_i), "city": city_i
        }
        price = predict_price_for_input(sample)
        st.success(f"💵 Precio estimado: **${price:,.0f} USD**")
