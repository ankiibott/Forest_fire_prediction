# FirePredict: Spatiotemporal Wildfire Risk Forecasting

![WhatsApp Image 2025-10-27 at 8 36 59 PM](https://github.com/user-attachments/assets/411ac497-1ef9-42d2-ab3e-5ca5db3fded3) ![WhatsApp Image 2025-10-27 at 8 37 26 PM](https://github.com/user-attachments/assets/3be94aa0-eae5-4c14-b314-b93b6f1ab261)


![Project Status](https://img.shields.io/badge/Status-In--Development-orange)
![Framework](https://img.shields.io/badge/Architecture-CNN--RNN--UNet-blue)
![Data](https://img.shields.io/badge/Datasets-ERA5%20|%20DEM%20|%20LULC-green)

A deep learning pipeline designed to predict forest fire probability grids for the Uttarakhand region. By integrating dynamic atmospheric variables with static topographical features, the system provides high-resolution risk maps to aid in disaster prevention.

---

## 🛠️ Project Overview

This project implements a hybrid deep learning approach to handle the complex nature of geospatial environmental data. We treat wildfire prediction as a spatiotemporal sequence problem, utilizing the following architecture:

* **CNN (Convolutional Neural Network):** Used for spatial feature extraction from raster (image-based) datasets including terrain and temperature gradients.
* **RNN (LSTM):** Used to process the temporal dependencies in the 6-hour input sequences.
* **U-Net Architecture:** Utilized to ensure the output fire probability grid maintains the exact spatial resolution and context as the input raster channels (Image-to-Image mapping).

## 📊 Data Pipeline & Preprocessing

The model ingests a combination of dynamic and static data channels:

1.  **ERA5 Atmospheric Data:** Hourly data for temperature (T2M), dew point (D2M), and precipitation.
2.  **DEM (Digital Elevation Model):** Provides topographic context (elevation/slope).
3.  **LULC (Land Use Land Cover):** * Processed into **Categorical** variables to identify fuel types (dense forest, shrubland, water, etc.).
    * Converted from tabular data to binary rasters for spatial alignment.

### Sliding Window Logic
To capture the evolution of fire conditions, we implemented a **Sliding Window** approach:
* **Input Window:** 6 hours of historical geospatial data.
* **Output Window:** 3 hours of future fire probability prediction.
* **Format:** All temporal data is serialized into a `.csv` structure to facilitate the sliding window generator before being reshaped into tensors for the model.

---

## 🖥️ System Modules

### 1. Geospatial Analysis Dashboard
The core interface allows for the upload of 7 geospatial channels (32 files total). It generates a **13×13 fire probability grid** with high-precision latitude/longitude bounds.

### 2. Interactive Fire Map
A Leaflet-powered interactive map that visualizes "High," "Medium," and "Low" risk zones across Uttarakhand, allowing users to toggle layers like Vegetation, Temperature, and Settlements.

### 3. AI Research Assistant
An integrated chat interface capable of exploring and querying uploaded **NetCDF** files, providing insights into the ocean and atmospheric data trends.

---

## 📂 Repository Structure

```text
├── Data_preprocessing/      # Tabular to binary raster conversion
├── LULC_categorical/        # Land use categorization scripts
├── Sequence_generation/     # Sliding window logic (6hr in -> 3hr out)
├── UI_page/                 # React/Vite based dashboard frontend
├── Time_bands/              # Normalization for raster_frp
├── VIIRS_vs_LULC.ipynb      # Correlation analysis of fire points
└── package-lock.json        # Frontend dependency management
