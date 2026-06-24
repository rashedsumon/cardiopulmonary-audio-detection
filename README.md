# Cardiopulmonary Sound Signals AI Classifier

An AI-powered web application deployed via Streamlit Cloud that analyzes cardiopulmonary audio signals to assist in classifying heart and lung sound anomalies.

## Setup Instructions

1. **Kaggle API Credentials**: 
   To allow `kagglehub` to download the dataset seamlessly on Streamlit Cloud, you must add your Kaggle credentials as **Secrets** in the Streamlit Cloud dashboard:
   ```toml
   KAGGLE_USERNAME = "your_kaggle_username"
   KAGGLE_KEY = "your_kaggle_api_key"