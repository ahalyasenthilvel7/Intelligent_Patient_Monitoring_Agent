# Intelligent Patient Monitoring Agent

An Agentic AI-powered patient monitoring prototype combining Machine Learning, real-time simulated vital signs, intelligent decision-making and automated alert generation.

## Project Overview

The Intelligent Patient Monitoring Agent continuously monitors simulated patient vital signs and uses trained Machine Learning models to identify unusual measurement patterns.

The Agentic AI workflow analyses model predictions, checks historical readings, determines alert priorities and generates notifications for human review.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest
- XGBoost
- LangGraph
- Streamlit
- Plotly

## Key Features

- Simulated real-time patient monitoring
- Heart rate and SpO2 monitoring
- Blood pressure and temperature monitoring
- Random Forest and XGBoost model comparison
- Patient history analysis
- Agentic decision-making workflow
- Automated alert generation
- Interactive Streamlit dashboard

## Agent Workflow

1. Collect patient readings.
2. Analyse readings using ML models.
3. Review previous patient measurements.
4. Determine alert priority.
5. Generate notifications.
6. Display results on the dashboard.

## Installation

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Important Disclaimer

This project uses synthetic data for educational and demonstration purposes. Its model predictions are not clinically validated and must not be used for medical diagnosis, treatment or real patient-care decisions.
