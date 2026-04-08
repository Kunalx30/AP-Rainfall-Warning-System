# 🌧️ AP Rainfall Early Warning System

### Agricultural Flood Risk Intelligence for Andhra Pradesh

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python\&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge\&logo=flask\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-orange?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![GeoPandas](https://img.shields.io/badge/GeoPandas-GIS%20Analysis-green?style=for-the-badge)
![Leaflet](https://img.shields.io/badge/Leaflet.js-Interactive%20Map-brightgreen?style=for-the-badge\&logo=leaflet\&logoColor=white)

🔴 **Live Application:**
https://ap-rainfall-warning-system.vercel.app/

</div>

---

## 📌 Overview

The **AP Rainfall Early Warning System** is a machine learning–driven web application designed to predict **village-level flood risk** across Andhra Pradesh.

The system integrates:

* Historical rainfall data (2021–2025)
* GIS infrastructure data (canals & embankments)
* Machine Learning (Random Forest)

It classifies villages into risk levels and supports **data-driven decision-making for disaster management and agriculture planning**.

---

## 🎯 Key Features

* ⚡ **Real-time Predictions** — Instant flood risk classification
* 🗺️ **Interactive GIS Map** — Village-level visualization using Leaflet
* 📊 **Analytical Dashboard** — Risk distribution and insights
* 🌊 **Infrastructure-Aware Predictions** — Considers proximity to canals & embankments
* 📱 **Responsive Design** — Optimized for mobile and desktop
* 🤖 **Model Confidence Scores** — Probability-based predictions
* 🚨 **Priority Risk Identification** — Highlights high-risk villages

---

## 🛠️ Technology Stack

| Layer               | Technologies                 |
| ------------------- | ---------------------------- |
| Data Processing     | Pandas, NumPy                |
| Geospatial Analysis | GeoPandas, Shapely           |
| Machine Learning    | Scikit-learn (Random Forest) |
| Backend             | Flask (REST API)             |
| Frontend            | Leaflet.js, Chart.js         |
| Storage             | Parquet (PyArrow)            |
| Deployment          | Vercel                       |

---

## 📁 Project Structure

```
AP_Warning_System/
│
├── app.py                      # Flask API
├── predictor.py                # ML logic
│
├── models/
│   ├── rf_model.pkl
│   ├── label_encoder.pkl
│   ├── feature_cols.pkl
│   ├── village_lookup.parquet
│   └── master_dataset.parquet
│
└── templates/
    └── index.html              # Frontend UI
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/yourusername/ap-rainfall-warning-system.git
cd ap-rainfall-warning-system

pip install flask flask-cors pandas numpy geopandas shapely scikit-learn pyarrow

python app.py
```

Access the application at:
http://localhost:5000

---

## 🚀 Usage

1. Open the application
2. Select a district
3. Enter rainfall data
4. Generate predictions
5. Analyze results on map and dashboard

---

## 📡 API Endpoints

| Method | Endpoint                  | Description         |
| ------ | ------------------------- | ------------------- |
| GET    | `/api/health`             | Health check        |
| GET    | `/api/districts`          | List districts      |
| GET    | `/api/mandals/<district>` | List mandals        |
| POST   | `/api/predict`            | Predict flood risk  |
| POST   | `/api/high_risk`          | High-risk villages  |
| GET    | `/api/stats/<district>`   | Historical insights |

---

## 📊 Dataset Information

| Dataset       | Source                                | Size            | Coverage       |
| ------------- | ------------------------------------- | --------------- | -------------- |
| Rainfall Data | IMD (India Meteorological Department) | 28.4M rows      | 2021–2025      |
| Canal Data    | AP Irrigation Department              | 16,408 segments | Andhra Pradesh |
| Embankments   | AP Irrigation Department              | 60 structures   | Andhra Pradesh |

**Key Highlights:**

* 220+ villages across 28 districts
* Geospatial transformation: WGS84 → UTM Zone 44N
* Integrated rainfall + GIS datasets

📁 Dataset & Notebook:
https://drive.google.com/drive/folders/1kqtmTKg7kaSwS_vjUwMb2I_o5XwqSsup?usp=drive_link

---

## 👨‍💻 Author

**Kunal Chandelkar**

Aspiring Data Analyst | Machine Learning Enthusiast
Focused on building real-world, data-driven solutions

> *"Data-driven decisions can save lives during flood emergencies."*

---

## 🔗 Project Links

* 🌐 Live Application:Live Application: https://ap-rainfall-warning-system.vercel.app/
* 📊 Dataset & Notebook: https://drive.google.com/drive/folders/1kqtmTKg7kaSwS_vjUwMb2I_o5XwqSsup?usp=drive_link

---

<sub>Built using Flask · Random Forest · GeoPandas · Leaflet.js</sub>
