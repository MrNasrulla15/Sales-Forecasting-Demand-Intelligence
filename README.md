# Sales Forecasting & Demand Intelligence Dashboard

A production-ready Streamlit application for exploring sales trends, generating forecasts, detecting anomalies, and segmenting demand patterns using the Superstore sales dataset.

## Overview

This project combines:
- a sales overview experience,
- a forecast explorer for short-term projections,
- an anomaly detection report,
- and a demand segmentation page.

The app uses shared Python modules under [src](src) so the dashboard and the notebook remain aligned without duplicating logic.

## Project Structure

```text
.
├── app.py                # Streamlit dashboard entry point
├── analysis.ipynb        # Notebook with analysis and visualizations
├── requirements.txt      # Python dependencies for local execution
├── data/                 # Raw and processed data files
│   ├── train.csv         # Main sales dataset
│   └── processed/        # Processed intermediate data files
├── charts/               # Exported chart images from the notebook
└── src/                  # Shared logic used by the notebook and app
    ├── config.py
    ├── data_loader.py
    ├── forecasting.py
    ├── anomaly.py
    └── segmentation.py
```

## Features

- Interactive overview of sales activity by time period and category
- Forecast exploration using the shared forecasting pipeline
- Anomaly detection via isolation forest and rolling z-score methods
- Demand segmentation and cluster-based interpretation

## Prerequisites

- Python 3.10+
- pip
- A virtual environment is recommended

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For Linux/macOS, use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Running the App

From the project root:

```bash
streamlit run app.py
```

The app will launch locally in the browser at the Streamlit URL shown in the terminal.

## Notes for GitHub Upload

This repository is prepared for GitHub upload as a standard Python project:
- source files are organized under [src](src),
- the app entry point is [app.py](app.py),
- dependencies are listed in [requirements.txt](requirements.txt),
- and the dataset is included in [train.csv](train.csv).

No files are uploaded to GitHub from this session.
