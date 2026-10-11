"""Streamlit demo. Run AFTER main.py has saved outputs/model.pkl:
    streamlit run app.py
"""
import json

import joblib
import streamlit as st

from src.config import OUT_DIR, CLASSES, BASELINE_DOSE_MM
from src.recommend import TYPICAL, predict_probs, recommend, explain

# 1. Load the logo image file from your root folder
from PIL import Image
logo_image = Image.open("logo.jpg")

# 2. Set page config FIRST (and swap out the droplet emoji for your logo!)
st.set_page_config(page_title="Adaptive Irrigation Intelligence", page_icon=logo_image, layout="wide")

# 3. Add it to your sidebar layout
st.sidebar.image(logo_image, use_container_width=True)

STAGES = ["Sowing", "Vegetative", "Flowering", "Harvest"]
SOILS = ["Sandy", "Clay", "Loamy", "Silt"]
CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Potato", "Sugarcane"]
SEASONS = ["Kharif", "Rabi", "Zaid"]
IRR_TYPES = ["Canal", "Drip", "Sprinkler", "Rainfed"]
SOURCES = ["Reservoir", "River", "Rainwater", "Groundwater"]
REGIONS = ["North", "South", "East", "West", "Central"]

# Demo scenarios (objective O6): low, moderate and high stress
SCENARIOS = {
    "Low stress": dict(soil_moisture=58.0, temperature_c=21.0, humidity=82.0, wind_speed_kmh=4.0,
                       rainfall_mm=1900.0, previous_irrigation_mm=70.0, crop_growth_stage="Vegetative"),
    "Moderate stress": dict(soil_moisture=32.0, temperature_c=30.0, humidity=55.0, wind_speed_kmh=11.0,
                            rainfall_mm=900.0, previous_irrigation_mm=50.0, crop_growth_stage="Flowering"),
    "High stress": dict(soil_moisture=12.0, temperature_c=39.0, humidity=30.0, wind_speed_kmh=17.0,
                        rainfall_mm=250.0, previous_irrigation_mm=10.0, crop_growth_stage="Flowering"),
}


@st.cache_resource
def load_bundle():
    path = OUT_DIR / "model.pkl"
    return joblib.load(path) if path.exists() else None


def set_scenario(name):
    for k, v in SCENARIOS[name].items():
        st.session_state[f"in_{k}"] = v


def init_state():
    for k, v in TYPICAL.items():
        st.session_state.setdefault(f"in_{k}", v)


bundle = load_bundle()
init_state()

st.title("💧 Adaptive Irrigation Intelligence")
st.caption("Predicts irrigation need from soil, weather and crop data, then turns it into a farmer-friendly "
           "recommendation. Decision-support only; combine with your own field observation.")

# AUTOMATIC TRAINING FALLBACK IF MODEL IS MISSING
if bundle is None:
    import os
    st.info("🔄 First-time setup: Training the machine learning model on the cloud server. Please wait about 30 seconds...")
    
    # Create the missing outputs directory and run the training pipeline script
    os.makedirs(str(OUT_DIR), exist_ok=True)
    os.system("python main.py --synthetic")
    
    # Reload the newly created model file weights
    st.cache_resource.clear()
    bundle = load_bundle()
    
    if bundle is None:
        st.error("Failed to auto-train the model. Please check your GitHub repository file layout.")
        st.stop()
    else:
        st.success("🎉 Training complete! Loading dashboard workspace...")
        st.rerun()

tab_adv, tab_res = st.tabs(["Advisor", "Model results"])

with tab_adv:
    st.write("**Try a demo scenario:**")
    cols = st.columns(len(SCENARIOS))
    for c, name in zip(cols, SCENARIOS):
        c.button(name, on_click=set_scenario, args=(name,), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Field conditions")
        st.slider("Soil moisture (%)", 8.0, 65.0, step=0.5, key="in_soil_moisture")
        st.slider("Temperature (°C)", 12.0, 42.0, step=0.5, key="in_temperature_c")
        st.slider("Humidity (%)", 25.0, 95.0, step=1.0, key="in_humidity")
        st.slider("Wind speed (km/h)", 0.5, 20.0, step=0.5, key="in_wind_speed_kmh")
        st.slider("Rainfall (mm)", 0.0, 2500.0, step=10.0, key="in_rainfall_mm")
        st.slider("Previous irrigation (mm)", 0.0, 120.0, step=1.0, key="in_previous_irrigation_mm")
        st.selectbox("Crop growth stage", STAGES, key="in_crop_growth_stage")
        st.selectbox("Soil type", SOILS, key="in_soil_type")
        st.selectbox("Crop", CROPS, key="in_crop_type")
        with st.expander("More field details (optional)"):
            st.slider("Soil pH", 4.8, 8.2, step=0.1, key="in_soil_ph")
            st.slider("Organic carbon", 0.3, 1.6, step=0.05, key="in_organic_carbon")
            st.slider("Electrical conductivity", 0.1, 3.5, step=0.05, key="in_electrical_conductivity")
            st.slider("Sunlight hours", 4.0, 11.0, step=0.5, key="in_sunlight_hours")
            st.slider("Field area (hectare)", 0.3, 15.0, step=0.1, key="in_field_area_hectare")
            st.selectbox("Season", SEASONS, key="in_season")
            st.selectbox("Irrigation type", IRR_TYPES, key="in_irrigation_type")
            st.selectbox("Water source", SOURCES, key="in_water_source")
            st.selectbox("Mulching used", ["No", "Yes"], key="in_mulching_used")
            st.selectbox("Region", REGIONS, key="in_region")

    row = {k: st.session_state[f"in_{k}"] for k in TYPICAL}
    probs = predict_probs(bundle, row)
    rec = recommend(probs, row)
    ci = CLASSES.index(rec["class"])

    with right:
        st.subheader("Recommendation")
        icon = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}[rec["class"]]
        st.markdown(f"### {icon} {rec['action']}")
        st.write(rec["advice"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted need", rec["class"], f"{rec['confidence']:.0%} confidence", delta_color="off")
        m2.metric("Suggested depth", f"{rec['dose_mm']} mm")
        m3.metric(f"vs fixed {BASELINE_DOSE_MM} mm", f"{rec['vs_fixed_mm']:+d} mm saved",
                  delta_color="off")
        for n in rec["notes"]:
            st.info(n)

        st.write("**Model probabilities**")
        st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)}, horizontal=True)

        st.write(f"**Why {rec['class']}?** (top inputs pushing toward this result)")
        for feat, val, d in explain(bundle, row, ci):
            arrow = "⬆️ raises" if d > 0 else "⬇️ lowers"
            name = feat.replace("_", " ")
            st.write(f"- **{name}** = {val} {arrow} the {rec['class']} probability by {abs(d):.0%}")

with tab_res:
    st.subheader(f"Final model: {bundle['model_name']}")
    comp = OUT_DIR / "model_comparison.csv"
    if comp.exists():
        import pandas as pd
        st.dataframe(pd.read_csv(comp), use_container_width=True)
    water = OUT_DIR / "water_simulation.json"
    if water.exists():
        w = json.loads(water.read_text())
        a, b, c = st.columns(3)
        a.metric("Water saved vs fixed schedule", f"{w['water_saved_vs_fixed_pct']:.1f}%")
        b.metric("Under-irrigated fields", f"{w['under_irrigated_pct']:.2f}%")
        c.metric("Over-irrigated fields", f"{w['over_irrigated_pct']:.2f}%")
        st.caption("Doses per class (10 / 30 / 60 mm) and the 30 mm fixed baseline are assumptions "
                   "set in config.py; the dataset is synthetic (Kaggle Playground S6E4).")
    for img in sorted(OUT_DIR.glob("*.png")):
        st.image(str(img), caption=img.stem.replace("_", " "), width=480)
