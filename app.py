from flask import Flask, jsonify, request ,render_template
from flask_cors import CORS
import os
from predictor import (
    predict_risk, get_summary,
    get_districts, get_mandals, get_villages,
    get_district_stats
)

app = Flask(__name__)
CORS(app)

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
FEATURED_PARQUET  = os.path.join(BASE_DIR, "models", "featured_dataset.parquet")


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "ok",
        "system" : "AP Rainfall Early Warning System",
        "version": "1.0"
    })


# ── Dropdowns ─────────────────────────────────────────────────────────────────
@app.route("/api/districts", methods=["GET"])
def districts():
    return jsonify(get_districts())


@app.route("/api/mandals/<path:district>", methods=["GET"])
def mandals(district):
    return jsonify(get_mandals(district))


@app.route("/api/villages/<path:district>/<path:mandal>", methods=["GET"])
def villages(district, mandal):
    return jsonify(get_villages(district, mandal))


# ── Main Prediction ───────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validate
    for field in ["district", "date", "rainfall_mm"]:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        results = predict_risk(
            district      = data["district"],
            date_str      = data["date"],
            rainfall_mm   = float(data["rainfall_mm"]),
            rainfall_3day = float(data["rainfall_3day"])  if "rainfall_3day"  in data else None,
            rainfall_7day = float(data["rainfall_7day"])  if "rainfall_7day"  in data else None,
            rainfall_30day= float(data["rainfall_30day"]) if "rainfall_30day" in data else None,
            mandal        = data.get("mandal")
        )

        return jsonify({
            "district": data["district"],
            "date"    : data["date"],
            "summary" : get_summary(results),
            "results" : results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── High Risk Only ────────────────────────────────────────────────────────────
@app.route("/api/high_risk", methods=["POST"])
def high_risk():
    data = request.get_json()
    try:
        results = predict_risk(
            district   = data["district"],
            date_str   = data["date"],
            rainfall_mm= float(data["rainfall_mm"])
        )
        filtered = [r for r in results if r["alert_level"] in ("RED", "ORANGE")]
        return jsonify({
            "district"       : data["district"],
            "date"           : data["date"],
            "high_risk_count": len(filtered),
            "results"        : filtered
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Historical Stats for Charts ───────────────────────────────────────────────
@app.route("/api/stats/<path:district>", methods=["GET"])
def stats(district):
    try:
        data = get_district_stats(district, FEATURED_PARQUET)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Serve Frontend ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 AP Rainfall Warning System starting...")
    print("   API running at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)