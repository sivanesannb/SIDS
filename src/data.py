"""Data loading, cleaning and a schema-matched synthetic fallback (for testing only)."""
import numpy as np
import pandas as pd
from .config import DATA_PATH, TARGET, RANDOM_STATE


def load_data(path=DATA_PATH, sample=None):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return _clean(df, sample)


def _clean(df, sample=None):
    df = df.drop(columns=[c for c in ["id"] if c in df.columns])
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    if sample and sample < len(df):
        df = df.sample(sample, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def make_synthetic(n=60000, seed=RANDOM_STATE):
    """Same schema/ranges as the Kaggle data. ONLY for smoke-testing the code
    when train.csv is not present. Never report results from this."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "soil_type": r.choice(["Sandy", "Clay", "Loamy", "Silt"], n),
        "soil_ph": r.uniform(4.8, 8.2, n),
        "soil_moisture": r.uniform(8, 65, n),
        "organic_carbon": r.uniform(0.3, 1.6, n),
        "electrical_conductivity": r.uniform(0.1, 3.5, n),
        "temperature_c": r.uniform(12, 42, n),
        "humidity": r.uniform(25, 95, n),
        "rainfall_mm": r.uniform(0.4, 2500, n),
        "sunlight_hours": r.uniform(4, 11, n),
        "wind_speed_kmh": r.uniform(0.5, 20, n),
        "crop_type": r.choice(["Sugarcane", "Wheat", "Rice", "Potato", "Cotton", "Maize"], n),
        "crop_growth_stage": r.choice(["Sowing", "Vegetative", "Flowering", "Harvest"], n),
        "season": r.choice(["Kharif", "Rabi", "Zaid"], n),
        "irrigation_type": r.choice(["Canal", "Drip", "Sprinkler", "Rainfed"], n),
        "water_source": r.choice(["Reservoir", "River", "Rainwater", "Groundwater"], n),
        "field_area_hectare": r.uniform(0.3, 15, n),
        "mulching_used": r.choice(["Yes", "No"], n),
        "previous_irrigation_mm": r.uniform(0, 120, n),
        "region": r.choice(["North", "South", "East", "West", "Central"], n),
    })
    score = (-0.12 * d.soil_moisture + 0.10 * d.temperature_c + 0.15 * d.wind_speed_kmh
             - 0.0006 * d.rainfall_mm + r.normal(0, 1.3, n))
    q_low, q_med = np.quantile(score, [0.587, 0.967])
    d["irrigation_need"] = np.where(score < q_low, "Low", np.where(score < q_med, "Medium", "High"))
    return d
