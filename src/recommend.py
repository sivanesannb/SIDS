"""Recommendation layer + simple local explanation.
Works with the bundle saved by main.py (outputs/model.pkl)."""
import pandas as pd

from .config import CLASSES, DOSE_MM, BASELINE_DOSE_MM
from .features import add_features

# Typical field values: used to fill inputs the user doesn't give, and as the
# "neutral" value when measuring how much each input pushes the prediction.
TYPICAL = {
    "soil_type": "Loamy", "soil_ph": 6.5, "soil_moisture": 35.0, "organic_carbon": 0.92,
    "electrical_conductivity": 1.74, "temperature_c": 28.0, "humidity": 60.0,
    "rainfall_mm": 1400.0, "sunlight_hours": 7.5, "wind_speed_kmh": 10.0,
    "crop_type": "Wheat", "crop_growth_stage": "Vegetative", "season": "Kharif",
    "irrigation_type": "Canal", "water_source": "Reservoir", "field_area_hectare": 7.5,
    "mulching_used": "No", "previous_irrigation_mm": 62.0, "region": "South",
}

# Crop-stage adjustment of the base dose (Flowering = most water-sensitive)
STAGE_FACTOR = {"Sowing": 0.8, "Vegetative": 1.0, "Flowering": 1.2, "Harvest": 0.5}

LABELS = {
    "No irrigation": "Soil has enough water. Skip irrigation and re-check in a day or two.",
    "Monitor": "Water level is adequate but not certain. Keep monitoring; light top-up only if soil dries.",
    "Irrigation recommended": "Moderate irrigation recommended. Apply the suggested dose.",
    "High water-stress risk": "Urgent: the crop is likely water-stressed. Irrigate now.",
}


def _frame(bundle, rows):
    """Rows (list of dicts) -> model-ready DataFrame (features added, training column order)."""
    df = pd.DataFrame([{**TYPICAL, **r} for r in rows])
    return add_features(df)[bundle["input_columns"]]


def predict_probs(bundle, row):
    return bundle["model"].predict_proba(_frame(bundle, [row]))[0]


def recommend(probs, row):
    """Predicted class + crop stage + previous irrigation -> action and dose."""
    i = int(max(range(len(probs)), key=lambda k: probs[k]))
    cls, conf = CLASSES[i], float(probs[i])
    stage = row.get("crop_growth_stage", TYPICAL["crop_growth_stage"])
    prev = float(row.get("previous_irrigation_mm", TYPICAL["previous_irrigation_mm"]))

    notes = []
    dose = DOSE_MM[cls] * STAGE_FACTOR.get(stage, 1.0)
    if stage == "Flowering":
        notes.append("Flowering is the most water-sensitive stage, so the dose is raised 20%.")
    elif stage == "Harvest":
        notes.append("Water demand falls near harvest, so the dose is halved.")
    elif stage == "Sowing":
        notes.append("Seeds rot if soil is too wet, so the dose is reduced 20%.")
    if prev >= 80 and cls != "High":
        dose *= 0.7
        notes.append(f"Heavy recent irrigation ({prev:.0f} mm), so the dose is cut 30%.")
    elif prev <= 20 and cls == "High":
        dose *= 1.15
        notes.append(f"Very little recent irrigation ({prev:.0f} mm), so the dose is raised 15%.")
    dose = int(round(dose / 5) * 5)

    if cls == "Low":
        action = "No irrigation" if conf >= 0.8 else "Monitor"
        if action == "No irrigation":
            dose = 0
    elif cls == "Medium":
        action = "Irrigation recommended"
    else:
        action = "High water-stress risk"
    if conf < 0.6:
        notes.append(f"Model confidence is only {conf:.0%}; confirm with a field check.")

    saved = BASELINE_DOSE_MM - dose
    return {"class": cls, "confidence": conf, "action": action, "advice": LABELS[action],
            "dose_mm": dose, "vs_fixed_mm": saved, "notes": notes}


def explain(bundle, row, class_idx, top=4):
    """Local explanation: for each input, swap it for its typical value and see how much
    the predicted class's probability changes. Positive = this input pushes toward the class."""
    full = {**TYPICAL, **row}
    base = predict_probs(bundle, full)[class_idx]
    keys = [k for k in TYPICAL if full[k] != TYPICAL[k]]
    if not keys:
        return []
    variants = [{**full, k: TYPICAL[k]} for k in keys]
    p = bundle["model"].predict_proba(_frame(bundle, variants))[:, class_idx]
    out = [(k, full[k], float(base - pk)) for k, pk in zip(keys, p)]
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out[:top]
