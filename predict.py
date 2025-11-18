import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_folium import st_folium
import folium
import json  # ⬅️ NEW

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
        precio_m2_barrio = pickle.load(f)

    with open("zona_premium_map_final.pkl", "rb") as f:
        barrio_zona = pickle.load(f)

    with open("xgb_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, kmeans, precio_m2_barrio, barrio_zona, feature_names

model, kmeans, precio_m2_barrio, barrio_zona, feature_names = load_artifacts()

barrios_conocidos = sorted(list(precio_m2_barrio.index))
barrios_conocidos.insert(0, "Desconocido")
property_types = ["Departamento", "PH", "Casa", "Casa de campo"]

# =========================================
# Función de ingeniería de features
# =========================================
def build_features_for_prediction(
    lat, lon, rooms, bedrooms, bathrooms,
    surface_total, surface_covered,
    property_type, barrio_l3
):
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

    global_med_full = precio_m2_barrio.median()
    X["precio_m2_barrio"] = X["l3"].map(precio_m2_barrio).fillna(global_med_full)
    X["zona_premium"] = X["l3"].map(barrio_zona).fillna(1).astype(int)
    cluster = kmeans.predict([[lat, lon]])[0]
    X["cluster_geo"] = cluster

    X_xgb = pd.get_dummies(X, columns=["property_type", "l3", "cluster_geo"], drop_first=True)

    for col in feature_names:
        if col not in X_xgb.columns:
            X_xgb[col] = 0

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
barrio_l3 = st.sidebar.selectbox("Barrio (l3)", barrios_conocidos)

# =========================
# Mapa con barrios.geojson
# =========================
st.subheader("Ubicación (clic en el mapa)")

# ⬅️ NEW: cargar GeoJSON de barrios
with open("barrios.geojson", "r", encoding="utf-8") as f:
    barrios_geojson = json.load(f)

mapa = folium.Map(location=[-34.60, -58.44], zoom_start=12)

# ⬅️ NEW: agregar capa con límites en negro
folium.GeoJson(
    barrios_geojson,
    name="Barrios CABA",
    style_function=lambda feature: {
        "color": "black",
        "weight": 1.5,
        "fillOpacity": 0,  # sin relleno, solo bordes
    },
).add_to(mapa)

folium.LayerControl().add_to(mapa)

mapa.add_child(folium.LatLngPopup())
map_result = st_folium(mapa, width=700, height=500)

st.markdown("### Seleccioná la ubicación en el mapa y completá los datos en la barra lateral.")

if st.button("🔮 Predecir precio"):
    if not map_result or not map_result.get("last_clicked"):
        st.error("Por favor, seleccioná una ubicación en el mapa.")
    elif surface_covered > surface_total:
        st.error("La superficie cubierta no puede ser mayor que la superficie total.")
    else:
        lat = map_result["last_clicked"]["lat"]
        lon = map_result["last_clicked"]["lng"]

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

        pred_price = model.predict(X_input)[0]
        pred_price_rounded = int(round(pred_price, -2))
        st.success(f"💰 Precio estimado: **USD {pred_price_rounded:,.0f}**")
        st.caption(
            "El valor es una estimación basada en el modelo entrenado con datos históricos.\n"
            "No reemplaza una tasación profesional."
        )
