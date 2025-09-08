# 🔥 Forest Fire Prediction (ISRO Hackathon 2025 – PS-01)

This repository contains the **backend workflow** for our **Forest Fire Risk Prediction model**, developed during the **ISRO Hackathon 2025**.  
Our team worked on **Problem Statement 01 (Forest Fire Risk Prediction)**, and we proudly secured a place in the **Top 10 teams across India** 🚀.  

Although we couldn’t make it to the Top 3 in the final group interviews, this project reflects a **robust research-driven pipeline** combining **remote sensing data + meteorological inputs + deep learning models**.

---

## 🛰️ Data Sources

We fetched multi-source geospatial and meteorological datasets:

- **ERA5 (CDS Portal)** → Meteorological reanalysis  
- **Bhuvan (ISRO)** → LULC, DEM, geospatial layers  
- **FIRMS (NASA)** → Active fire hotspots  

---

## ⚡ Workflow Overview

### 🔹 1. Data Preprocessing  
- Normalization & cleaning (`data_cleaning/`)  
- LULC categorical encoding (`LULC_categorical/`)  
- DEM–LULC alignment (`Alignment/`)  

### 🔹 2. Sequence Generation  
- Implemented **sliding window** time-series:  
  - Input → **6 hours**  
  - Output → **1, 2, 3 hours**  
- Shifted 1-hr step → created dense sequences  
- Scripts: `time_sequences_csv.ipynb`, `Viirs_time_series.ipynb`, `era5_time_series.ipynb`  

### 🔹 3. Model Building (CNN – Multi-task Learning)  
- CNN extracts spatio-temporal features  
- Dual-head output:  
  - 🔥 `cls_out` → Fire / No Fire (binary classification)  
  - 📈 `reg_out` → Fire spread intensity (regression)  

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss={
        "reg_out": tf.keras.losses.MeanSquaredError(),
        "cls_out": tf.keras.losses.BinaryCrossentropy(from_logits=False),
    },
    loss_weights={"reg_out": 1.0, "cls_out": 0.3},
    metrics={
        "reg_out": [tf.keras.metrics.MeanAbsoluteError()],
        "cls_out": [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    },
)

flowchart TD
    A[🌍 Data Sources <br> ERA5 · Bhuvan · FIRMS] --> B[🛠 Data Preprocessing <br> Cleaning · LULC Encoding · Alignment]
    B --> C[⏳ Sequence Generation <br> Sliding Window (6h → 1/2/3h)]
    C --> D[🧠 CNN Model <br> Feature Extraction]
    D --> E1[🔥 Classification Head <br> Fire / No Fire]
    D --> E2[📈 Regression Head <br> Fire Intensity]
    E1 & E2 --> F[✅ Predictions <br> Risk Maps & Values]
