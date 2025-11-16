# predict.py
# App de Streamlit para predecir precios de propiedades en CABA
# usando el modelo XGBoost entrenado en el TP.

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================
# Carga de artefactos (cacheados)
# =========================================
@st.cache_resource
def load_artifacts():
    with open("modelo_xgboost_final.pkl", "rb") as f:
        model = pickle.load(f)

    with open("kmeans_final.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("precio_m2_barrio_final.pkl", "rb") as f:
        precio_m2_barrio = pickle.load(f)  # Series: index = l3 (barrio)

    with open("zona_premium_map_final.pkl", "rb") as f:
        barrio_zona = pickle.load(f)  # Series: index = l3 (barrio) → 0..3

    with open("xgb_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)  # lista de columnas usadas en el entrenamiento

    return model, kmeans, precio_m2_barrio, barrio_zona, feature_names


model, kmeans, precio_m2_barrio, barrio_zona, feature_names = load_artifacts()

# Lista de barrios conocidos (desde los índices de la serie)
barrios_conocidos = sorted(list(precio_m2_barrio.index))
barrios_conocidos.insert(0, "Desconocido")

# Tipos de propiedad principales usados en el entrenamiento
property_types = ["Departamento", "PH", "Casa", "Casa de campo"]

# =========================================
# Función de ingeniería de features
# =========================================
def build_features_for_prediction(
    lat, lon, rooms, bedrooms, bathrooms,
    surface_total, surface_covered,
    property_type, barrio_l3
):
    # 1) DataFrame base con las mismas columnas que en el entrenamiento
    data = {
        "lat": [lat],
        "lon": [lon],
        "rooms": [rooms],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "surface_total": [surface_total],
        "surface_covered": [surface_covered],
        "property_type": [property_type],
        "l3": [barrio_l3],
    }

    X = pd.DataFrame(data)

    # 2) Feature: precio_m2_barrio (solo mapeo, NO usa el precio actual)
    global_med_full = precio_m2_barrio.median()
    X["precio_m2_barrio"] = (
        X["l3"].map(precio_m2_barrio)
               .fillna(global_med_full)
    )

    # 3) Feature: zona_premium (0..3, o 1 si el barrio es desconocido)
    X["zona_premium"] = (
        X["l3"].map(barrio_zona)
               .fillna(1)
               .astype(int)
    )

    # 4) Feature: cluster_geo con KMeans sobre lat / lon
    cluster = kmeans.predict([[lat, lon]])[0]
    X["cluster_geo"] = cluster

    # 5) One-hot encoding igual que en el entrenamiento
    X_xgb = pd.get_dummies(
        X,
        columns=["property_type", "l3", "cluster_geo"],
        drop_first=True
    )

    # 6) Alinear columnas con las usadas por el modelo
    for col in feature_names:
        if col not in X_xgb.columns:
            X_xgb[col] = 0

    # Eliminar cualquier columna extra que no esté en feature_names
    X_xgb = X_xgb[feature_names]

    return X_xgb


# =========================================
# Interfaz Streamlit
# =========================================
st.set_page_config(page_title="Predicción de precios CABA", layout="centered")

st.title("🧠 Predicción de precio de propiedades en CABA")
st.markdown(
    "App demo del TP de Programación / Data Science.\n"
    "Modelo: **XGBoost** entrenado sobre datos de Properati (CABA)."
)

st.sidebar.header("Parámetros de la propiedad")

# Inputs geográficos
st.sidebar.subheader("Ubicación (coordenadas)")
lat = st.sidebar.number_input(
    "Latitud",
    value=-34.60,
    format="%.6f",
    help="Latitud en grados decimales (ej: -34.60 para CABA).",
)
lon = st.sidebar.number_input(
    "Longitud",
    value=-58.44,
    format="%.6f",
    help="Longitud en grados decimales (ej: -58.44 para CABA).",
)

# Inputs físicos
st.sidebar.subheader("Características físicas")
rooms = st.sidebar.number_input("Ambientes (rooms)", min_value=0, max_value=20, value=3, step=1)
bedrooms = st.sidebar.number_input("Dormitorios (bedrooms)", min_value=0, max_value=20, value=2, step=1)
bathrooms = st.sidebar.number_input("Baños (bathrooms)", min_value=0, max_value=10, value=1, step=1)

surface_total = st.sidebar.number_input("Superficie total (m²)", min_value=1.0, max_value=2000.0, value=60.0, step=1.0)
surface_covered = st.sidebar.number_input("Superficie cubierta (m²)", min_value=0.0, max_value=2000.0, value=55.0, step=1.0)

# Inputs categóricos
st.sidebar.subheader("Tipo y barrio")
property_type = st.sidebar.selectbox("Tipo de propiedad", property_types)

barrio_l3 = st.sidebar.selectbox(
    "Barrio (l3)",
    barrios_conocidos,
    help="Barrios según Properati. Si no sabés, elegí 'Desconocido'."
)

st.markdown("### Ingrese los datos en la barra lateral y haga clic en **Predecir precio**.")

if st.button("🔮 Predecir precio"):
    # Validación simple
    if surface_covered > surface_total:
        st.error("La superficie cubierta no puede ser mayor que la superficie total.")
    else:
        # Construimos las features
        X_input = build_features_for_prediction(
            lat=lat,
            lon=lon,
            rooms=rooms,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            surface_total=surface_total,
            surface_covered=surface_covered,
            property_type=property_type,
            barrio_l3=barrio_l3,
        )

        # Predicción
        pred_price = model.predict(X_input)[0]

        # Formateo
        pred_price_rounded = int(round(pred_price, -2))  # redondeamos a la centena
        st.success(f"💰 Precio estimado: **USD {pred_price_rounded:,.0f}**")

        # Info adicional
        st.caption(
            "El valor es una estimación basada en el modelo entrenado con datos históricos.\n"
            "No reemplaza una tasación profesional."
        )
