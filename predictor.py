import pandas as pd
import numpy as np
import pickle
import os

# ─── Load Models ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Loading models...")

with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

village_lookup = pd.read_parquet(
    os.path.join(MODEL_DIR, "village_lookup.parquet")
)

print(f"✅ Models loaded | Villages in lookup: {len(village_lookup):,}")


# ─── Constants ─────────────────────────────────────────────────────────────────
ALERT_META = {
    "GREEN" : {
        "color"  : "#2ecc71",
        "bg"     : "#eafaf1",
        "emoji"  : "🟢",
        "label"  : "No Risk",
        "message": "No significant flood risk",
        "priority": 4
    },
    "YELLOW": {
        "color"  : "#f39c12",
        "bg"     : "#fef9e7",
        "emoji"  : "🟡",
        "label"  : "Moderate Risk",
        "message": "Monitor water levels — moderate risk",
        "priority": 3
    },
    "ORANGE": {
        "color"  : "#e67e22",
        "bg"     : "#fdf2e9",
        "emoji"  : "🟠",
        "label"  : "High Risk",
        "message": "Prepare evacuation — high flood risk",
        "priority": 2
    },
    "RED"   : {
        "color"  : "#e74c3c",
        "bg"     : "#fdedec",
        "emoji"  : "🔴",
        "label"  : "Extreme Risk",
        "message": "Immediate action required — extreme flood risk",
        "priority": 1
    },
}

RISK_ORDER = {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}


# ─── Rainfall Classification ───────────────────────────────────────────────────
def classify_rainfall(mm):
    if mm == 0:        return "No Rain"
    elif mm < 2.5:     return "Light"
    elif mm < 15.6:    return "Moderate"
    elif mm < 64.5:    return "Heavy"
    elif mm < 115.6:   return "Very Heavy"
    else:              return "Extremely Heavy"


# ─── Dropdowns ────────────────────────────────────────────────────────────────
def get_districts():
    return sorted(village_lookup["district"].unique().tolist())


def get_mandals(district):
    df = village_lookup[village_lookup["district"] == district]
    return sorted(df["mandal"].unique().tolist())


def get_villages(district, mandal):
    df = village_lookup[
        (village_lookup["district"] == district) &
        (village_lookup["mandal"]   == mandal)
    ]
    return sorted(df["village"].unique().tolist())


# ─── Core Prediction ──────────────────────────────────────────────────────────
def predict_risk(district, date_str, rainfall_mm,
                 rainfall_3day=None, rainfall_7day=None,
                 rainfall_30day=None, mandal=None):

    date        = pd.to_datetime(date_str)
    month       = date.month
    day_of_year = date.dayofyear
    is_monsoon  = 1 if 6 <= month <= 10 else 0

    # Smart defaults for accumulations if not provided
    if rainfall_3day  is None: rainfall_3day  = round(rainfall_mm * 1.5, 1)
    if rainfall_7day  is None: rainfall_7day  = round(rainfall_mm * 2.5, 1)
    if rainfall_30day is None: rainfall_30day = round(rainfall_mm * 4.0, 1)

    rainfall_anomaly = round(rainfall_mm - (rainfall_30day / 30), 2)

    # Filter villages
    mask = village_lookup["district"] == district
    if mandal:
        mask = mask & (village_lookup["mandal"] == mandal)

    villages = village_lookup[mask].copy().reset_index(drop=True)

    if villages.empty:
        return []

    # Attach rainfall features
    villages["rainfall_mm"]      = rainfall_mm
    villages["rainfall_3day"]    = rainfall_3day
    villages["rainfall_7day"]    = rainfall_7day
    villages["rainfall_30day"]   = rainfall_30day
    villages["rainfall_anomaly"] = rainfall_anomaly
    villages["month"]            = month
    villages["day_of_year"]      = day_of_year
    villages["is_monsoon"]       = is_monsoon

    X           = villages[FEATURE_COLS].fillna(0)
    y_pred      = rf_model.predict(X)
    y_proba     = rf_model.predict_proba(X)
    alert_labels= le.inverse_transform(y_pred)
    classes     = list(le.classes_)

    results = []
    for i, row in villages.iterrows():
        alert      = alert_labels[i]
        meta       = ALERT_META[alert]
        proba      = y_proba[i]
        confidence = round(float(proba[y_pred[i]]) * 100, 1)

        has_embankment = float(row["dist_embankment_km"]) < 50.0

        results.append({
            # Location
            "district"                  : row["district"],
            "mandal"                    : row["mandal"],
            "village"                   : row["village"],
            "latitude"                  : round(float(row["centroid_lat"]), 6),
            "longitude"                 : round(float(row["centroid_lon"]), 6),

            # Rainfall
            "rainfall_mm"               : rainfall_mm,
            "rainfall_category"         : classify_rainfall(rainfall_mm),
            "rainfall_3day"             : rainfall_3day,
            "rainfall_7day"             : rainfall_7day,
            "rainfall_30day"            : rainfall_30day,

            # Infrastructure
            "dist_canal_km"             : round(float(row["dist_canal_km"]), 2),
            "nearest_canal_name"        : str(row["nearest_canal_name"]) if "nearest_canal_name" in row.index and pd.notna(row["nearest_canal_name"]) else "N/A",
            "dist_embankment_km"        : round(float(row["dist_embankment_km"]), 2),
            "nearest_embankment_name"   : str(row["nearest_embankment_name"]) if "nearest_embankment_name" in row.index and pd.notna(row["nearest_embankment_name"]) else "N/A",
            "canal_proximity_score"     : int(row["canal_proximity_score"]),
            "embankment_proximity_score": int(row["embankment_proximity_score"]),
            "has_embankment_nearby"     : has_embankment,

            # Alert
            "alert_level"               : alert,
            "alert_color"               : meta["color"],
            "alert_bg"                  : meta["bg"],
            "alert_emoji"               : meta["emoji"],
            "alert_label"               : meta["label"],
            "alert_message"             : meta["message"],
            "confidence_pct"            : confidence,

            # Probabilities
            "prob_green"                : round(float(proba[classes.index("GREEN")])  * 100, 1),
            "prob_yellow"               : round(float(proba[classes.index("YELLOW")]) * 100, 1),
            "prob_orange"               : round(float(proba[classes.index("ORANGE")]) * 100, 1),
            "prob_red"                  : round(float(proba[classes.index("RED")])    * 100, 1),
        })

    # Sort by risk level — RED first
    results.sort(key=lambda x: RISK_ORDER[x["alert_level"]])
    return results


# ─── Summary ──────────────────────────────────────────────────────────────────
def get_summary(results):
    summary = {
        "total" : len(results),
        "RED"   : 0,
        "ORANGE": 0,
        "YELLOW": 0,
        "GREEN" : 0,
    }
    for r in results:
        summary[r["alert_level"]] += 1

    if summary["RED"]    > 0: summary["overall_alert"] = "RED"
    elif summary["ORANGE"]>0: summary["overall_alert"] = "ORANGE"
    elif summary["YELLOW"]>0: summary["overall_alert"] = "YELLOW"
    else:                     summary["overall_alert"] = "GREEN"

    return summary


# ─── Historical Stats (for graphs) ───────────────────────────────────────────
def get_district_stats(district, featured_parquet_path):
    df = pd.read_parquet(featured_parquet_path,
                         filters=[("district", "==", district)])
    df["date"] = pd.to_datetime(df["date"])

    monthly = (
        df.groupby(["year", "month"])["rainfall_mm"]
        .mean()
        .reset_index()
        .rename(columns={"rainfall_mm": "avg_rainfall"})
    )
    monthly["avg_rainfall"] = monthly["avg_rainfall"].round(2)

    alert_yearly = (
        df.groupby(["year", "alert_level"])
        .size()
        .reset_index(name="count")
    )

    top_villages = (
        df.groupby(["mandal", "village"])["rainfall_mm"]
        .max()
        .reset_index()
        .sort_values("rainfall_mm", ascending=False)
        .head(10)
    )

    return {
        "monthly_avg"  : monthly.to_dict(orient="records"),
        "alert_yearly" : alert_yearly.to_dict(orient="records"),
        "top_villages" : top_villages.to_dict(orient="records"),
    }