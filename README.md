# XFiRE 
### Satellite-Based Fire Monitoring, MODIS–VIIRS Harmonization & Machine Learning Forecasting

**XFiRE (eXplainable Fire Intelligence and Risk Estimation)** is a Python-based geospatial and machine-learning framework designed to process satellite-derived active-fire observations, harmonize MODIS and VIIRS datasets, analyse spatial and temporal fire activity, and generate multi-horizon weekly fire-activity forecasts.

The project integrates satellite remote sensing, statistical harmonization, machine learning, and geospatial analysis into a unified workflow.

---

## Overview

Satellite-based fire monitoring relies on observations from different sensors. However, differences in spatial resolution, detection sensitivity, and observation characteristics can introduce inconsistencies when combining long-term MODIS and VIIRS records.

XFiRE addresses this challenge through a structured processing pipeline that transforms raw NASA FIRMS observations into harmonized fire-activity datasets, spatial-temporal analyses, and forecasting outputs.

The framework focuses on two independent analytical tasks:

1. **Satellite Data Harmonization:** Improving comparability between MODIS and VIIRS active-fire observations.
2. **Fire Activity Forecasting:** Predicting future weekly satellite-observed fire activity using statistical and machine-learning approaches.

## Key Features

- **Multi-Sensor Data Integration:** Processes MODIS and VIIRS active-fire observations from NASA FIRMS.
- **Data Harmonization:** Spatial and temporal alignment of observations from different satellite sensors.
- **XGBoost Harmonization:** Machine-learning-based sensor harmonization evaluated against ratio-based baselines.
- **Spatial-Temporal Analysis:** Examines fire activity across geographic grids and time periods.
- **Weekly Forecasting:** Generates forecasts across multiple horizons, up to 26 weeks.
- **LightGBM Forecasting:** Uses gradient-boosted decision trees for weekly fire-activity prediction.
- **Hyperparameter Optimization:** Integrates Hyperopt with the Tree-structured Parzen Estimator (TPE).
- **Baseline Comparison:** Evaluates machine-learning models against climatology and seasonal-naive forecasting.
- **Uncertainty Estimation:** Provides forecast intervals and coverage diagnostics.
- **Interactive Dashboard:** Presents historical observations, harmonization results, spatial patterns, and forecasts.
- **Data Quality Diagnostics:** Identifies potential data issues and evaluates model reliability.

## System Architecture

```text
NASA FIRMS
    |
    v
MODIS + VIIRS Active-Fire CSV Data
    |
    v
Data Ingestion & Quality Control
    |
    v
Spatial Grid Aggregation
    |
    v
Temporal Alignment
    |
    v
MODIS–VIIRS Harmonization
    |
    +----------------------------+
    |                            |
    v                            v
Ratio-Based Baselines        XGBoost
    |                            |
    +-------------+--------------+
                  |
                  v
        Harmonization Evaluation
                  |
                  v
        Harmonized Fire Activity
                  |
                  v
        Weekly Feature Engineering
                  |
                  v
       LightGBM Forecasting
                  |
                  v
        Hyperopt (TPE) Tuning
                  |
                  v
       Baseline Model Comparison
                  |
                  v
       Multi-Horizon Forecasts
                  |
                  v
       Dashboard & Diagnostics
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Harmonization | XGBoost |
| Forecasting | LightGBM |
| Hyperparameter Optimization | Hyperopt (TPE) |
| Satellite Data | NASA FIRMS MODIS and VIIRS |
| Visualization | JavaScript, HTML, CSS |
| Analysis | Geospatial and statistical methods |

## Data Sources

XFiRE uses active-fire observations from NASA's Fire Information for Resource Management System (FIRMS).

- **MODIS:** Terra and Aqua satellite observations.
- **VIIRS:** Visible Infrared Imaging Radiometer Suite observations.

Data can include latitude, longitude, acquisition date, fire radiative power (FRP), confidence, satellite, and other available observation attributes.

**Data source:** https://firms.modaps.eosdis.nasa.gov/

## Input Data Structure

Place FIRMS CSV files in the following directories:

```text
xfire/
├── data/
│   ├── modis/
│   │   ├── modis_1.csv
│   │   └── modis_2.csv
│   │
│   └── viirs/
│       ├── viirs_1.csv
│       └── viirs_2.csv
│
├── config/
├── output/
└── ...
```

The standard input fields include:

```text
latitude
longitude
acq_date
frp
confidence
```

Additional fields such as `type`, `satellite`, `daynight`, and `acq_time` are used when available.

Input files should follow the expected FIRMS CSV format. The current directory-based ingestion expects lowercase `.csv` files directly inside the sensor folders.

## Harmonization Methodology

XFiRE evaluates alternative approaches for reducing differences between MODIS and VIIRS observations.

The harmonization workflow includes:

- Spatial aggregation into common grid cells.
- Temporal alignment of observations.
- Sensor-overlap analysis.
- Ratio-based harmonization baselines.
- XGBoost-based harmonization.
- Spatial holdout and chronological evaluation.

The objective is to improve comparability without assuming that the two sensors have identical detection characteristics.

## Forecasting Methodology

The forecasting component models historical weekly fire activity and estimates future activity across multiple prediction horizons.

**LightGBM**

A gradient-boosting framework used as the primary machine-learning forecasting candidate.

**Hyperopt**

Uses TPE-based hyperparameter optimization to search for improved model configurations.

**Baseline Models**

Forecasting performance is compared against simpler alternatives, including:

- Historical climatology.
- Seasonal-naive forecasting.

**Validation**

Chronological validation is used to evaluate forecasting performance while limiting information leakage from future observations.

Forecast quality is assessed using error metrics, bias, baseline comparisons, and prediction-interval coverage.

## Forecast Outputs

The framework is designed to produce:

- Weekly fire-activity forecasts.
- Multiple forecast horizons.
- Spatially aggregated predictions.
- Model comparison results.
- Forecast uncertainty intervals.
- Evaluation and data-quality reports.

Forecasts represent estimated satellite-observed fire activity, not confirmed future fire incidents or direct burned-area measurements.

## Running the Project

The framework is designed to run locally with Python.

After preparing the environment and placing the FIRMS CSV files in the appropriate directories, use the project's provided PowerShell launcher or CLI instructions.

See the project documentation for installation requirements, configuration options, and execution commands.

## Scientific Evaluation

XFiRE emphasizes measurable model performance rather than model complexity.

The evaluation framework considers:

- Harmonization accuracy.
- Spatial generalization.
- Chronological forecasting performance.
- Comparison against statistical baselines.
- Forecast bias.
- Prediction interval coverage.
- Data quality and observation gaps.

Machine-learning models are treated as candidates and are not assumed to outperform simpler statistical approaches.

## Project Limitations

- Satellite active-fire detections are not equivalent to individual fire incidents.
- Detection counts depend on sensor characteristics and observation conditions.
- Missing satellite observations can affect historical time series.
- VIIRS and MODIS observations require careful harmonization.
- Forecast accuracy depends on data coverage, historical patterns, and validation results.
- Forecasts should not be interpreted as guaranteed fire-event predictions.

## Project Vision

The long-term objective of XFiRE is to develop a scalable and scientifically evaluated satellite-based fire intelligence framework that connects historical fire observations, multi-sensor harmonization, spatial-temporal analysis, and machine-learning forecasting.

The framework is intended to support environmental research, satellite-based fire monitoring, and the development of future fire-activity early-warning and decision-support systems.

---

**Project:** XFiRE 
**Category:** Remote Sensing | Geospatial Analysis | Machine Learning | Environmental Monitoring  
**Data Source:** NASA FIRMS  
**Harmonization:** XGBoost and Ratio-Based Methods  
**Forecasting:** LightGBM and Hyperopt (TPE)  
**Forecast Horizon:** Up to 26 Weeks

**Status:** Research and development.

