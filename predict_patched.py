"""Interactive irrigation advisor.
Run AFTER main.py has trained and saved the model:   python predict.py
Press Enter at any question to accept the [default]."""
import joblib
import pandas as pd

from src.config import OUT_DIR, DOSE_MM
from src.features import add_features

MODEL_PATH = OUT_DIR / "model.pkl"   # saved by main.py as a dict bundle

# (column, question, default, allowed choices or None for a number)
KEY_INPUTS = [
    ("soil_moisture",     "Soil moisture % (8-65)",            35.0, None),
    ("temperature_c",     "Temperature in C (12-42)",          28.0, None),
    ("humidity",          "Humidity % (25-95)",                60.0, None),
    ("wind_speed_kmh",    "Wind speed km/h (0.5-20)",          10.0, None),
    ("rainfall_mm",       "Rainfall mm (0-2500)",            1400.0, None),
    ("crop_growth_stage", "Crop stage",                 "Vegetative", ["Sowing", "Vegetative", "Flowering", "Harvest"]),
    ("soil_type",         "Soil type",                       "Loamy", ["Sandy", "Clay", "Loamy", "Silt"]),
    ("crop_type",         "Crop",                            "Wheat", ["Rice", "Wheat", "Maize", "Cotton", "Potato", "Sugarcane"]),
]

# Less important fields: filled with typical values from the dataset
DEFAULTS = {
    "soil_ph": 6.5, "organic_carbon": 0.92, "electrical_conductivity": 1.74,
    "sunlight_hours": 7.5, "field_area_hectare": 7.5, "previous_irrigation_mm": 62.0,
    "season": "Kharif", "irrigation_type": "Canal", "water_source": "Reservoir",
    "mulching_used": "No", "region": "South",
}

ADVICE = {
    "Low": "Soil has enough water. Skip irrigation now and re-check in a day or two.",
    "Medium": "Moderate irrigation recommended. Apply a standard dose.",
    "High": "Urgent: crop is likely water-stressed. Irrigate now with a larger dose.",
}


def ask(col, text, default, choices):
    while True:
        shown = f"{text} [{default}]" + (f" options: {'/'.join(choices)}" if choices else "")
        raw = input(shown + ": ").strip()
        if raw == "":
            return default
        if choices:
            match = [c for c in choices if c.lower() == raw.lower()]
            if match:
                return match[0]
            print("  Please choose one of the listed options.")
        else:
            try:
                return float(raw)
            except ValueError:
                print("  Please enter a number.")


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run  python main.py  first.")
    bundle = joblib.load(MODEL_PATH)
    model, classes = bundle["model"], bundle["classes"]

    print(f"\n=== Smart Irrigation Advisor (model: {bundle['model_name']}) ===")
    row = dict(DEFAULTS)
    for col, text, default, choices in KEY_INPUTS:
        row[col] = ask(col, text, default, choices)

    X = add_features(pd.DataFrame([row]))
    X = X[bundle["input_columns"]]   # same columns and order as training
    probs = model.predict_proba(X)[0]
    pred = classes[int(probs.argmax())]

    print(f"\nRecommendation: {pred.upper()} irrigation need")
    print(ADVICE[pred])
    print(f"Suggested depth (assumed): ~{DOSE_MM[pred]} mm")
    print("Confidence: " + ", ".join(f"{c} {p:.0%}" for c, p in zip(classes, probs)))
    print("Note: decision-support only. Combine with your own field observation.")


if __name__ == "__main__":
    main()
