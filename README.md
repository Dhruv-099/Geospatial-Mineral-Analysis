# Mineral Deposit Classifier

A Streamlit application for geochemical analysis and machine learning classification of mineral deposit types based on trace element data.
**[Live Application](https://geospatial-mineral-analysis.streamlit.app/)**

## Features

*   **Data Analysis:** Analyze the elemental composition of individual deposit types using histograms, box plots, and pie charts.
*   **Deposit Comparison:** Compare the distribution of specific elements across all deposit types to identify key differentiators.
*   **Predict Deposit Type:** A real-time prediction interface that uses a trained PyCaret model to classify new samples from user-inputted elemental data.

## Technology Stack

*   **Analysis:** Python, Pandas, NumPy
*   **Dashboard:** Streamlit
*   **Visualization:** Plotly
*   **ML Pipeline:** PyCaret, Scikit-learn

## Setup and Usage

**1. Clone the repository:**```bash
git clone https://github.com/Dhruv-099/Geospatial-Mineral-Analysis.git
cd Geospatial-Mineral-Analysis
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Run the application:**
```bash
streamlit run app.py
```

## Project Workflow

1.  **Data Cleaning:** Ingested and cleaned five raw CSVs with inconsistent formatting into a single `master_mineral_data.csv`.
2.  **EDA:** Built the interactive Streamlit dashboard for data exploration and visualization.
3.  **Model Training:** Used PyCaret to automatically train, tune, and evaluate multiple classification models, selecting the best performer.
4.  **Integration:** Saved the final model pipeline and integrated it into the Streamlit app for live predictions.