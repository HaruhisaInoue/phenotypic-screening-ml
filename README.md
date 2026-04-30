# phenotypic-screening-ml
Phenotypic Compound Screening Using Machine Learning
Overview
This project performs phenotypic compound screening using machine learning based on MEA data from iPSC-derived neurons.
This repository provides a trained model for predicting compound efficacy from neuronal phenotypes.
Method
•	Input: MEA recordings from iPSC-derived neurons
•	Features: Phenotypic features of neuronal activity
•	Model: XGBoost
•	Task: Predict compound efficacy
Model
This repository includes a trained XGBoost model:
•	xgb_model_WHHCC_NB_active5.json
The model was trained on phenotypic features derived from MEA data.
Usage
Load the model in Python using XGBoost:
import xgboost as xgb

model = xgb.Booster()
model.load_model("xgb_model_WHHCC_NB_active5.json")
Data
MEA data is not included in this repository.
Author
Haruhisa Inoue
