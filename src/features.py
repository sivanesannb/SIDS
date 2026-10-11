"""Feature engineering: agronomy-informed features + encoding pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

STAGE_ORDER = ["Sowing", "Vegetative", "Flowering", "Harvest"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Evaporative demand proxy: hot, dry, windy, sunny air pulls water from soil
    d["vpd_proxy"] = (100 - d["humidity"]) * d["temperature_c"] / 100
    d["evap_demand"] = d["temperature_c"] * d["sunlight_hours"] * (1 + d["wind_speed_kmh"] / 20)
    # Water balance: how much water is in the soil relative to what is leaving
    d["moisture_deficit"] = 100 - d["soil_moisture"]
    d["moisture_per_demand"] = d["soil_moisture"] / (d["evap_demand"] + 1)
    d["rain_minus_demand"] = d["rainfall_mm"] / 100 - d["evap_demand"] / 10
    # Soil water-holding proxy
    d["retention_proxy"] = d["organic_carbon"] * d["soil_moisture"]
    d["total_water_recent"] = d["previous_irrigation_mm"] + d["rainfall_mm"] / 12
    # Crop-stage sensitivity (flowering is the most water-sensitive)
    d["stage_idx"] = d["crop_growth_stage"].map({s: i for i, s in enumerate(STAGE_ORDER)})
    d["flowering_flag"] = (d["crop_growth_stage"] == "Flowering").astype(int)
    d["temp_x_wind"] = d["temperature_c"] * d["wind_speed_kmh"]
    return d


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    return pre, cat_cols, num_cols
