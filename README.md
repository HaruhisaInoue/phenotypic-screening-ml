# Hypothesis-free multidimensional analysis of neuronal network activity enables convergent phenotypic compound screening

This repository contains the code, trained XGBoost model, and representative processed multielectrode array (MEA) data associated with the study *“Hypothesis-free multidimensional analysis of neuronal network activity enables convergent phenotypic compound screening.”*

## Repository contents

```text
.
├── README.md
├── preprocess.py
├── filter.csv
├── learn_and_predict.py
├── xgb_model.json
├── representative_train_data.csv
└── representative_test_data.csv
```

- `preprocess.py`: Preprocessing and data-cleaning procedures applied to the MEA feature data.
- `filter.csv`: Feature-selection and filtering information used by the preprocessing script.
- `learn_and_predict.py`: XGBoost model training, evaluation, and prediction workflow.
- `xgb_model.json`: Trained XGBoost model used for prediction.
- `representative_train_data.csv`: Representative subset of the processed Batch 1 data from healthy control (HC) and autism spectrum disorder (ASD) samples.
- `representative_test_data.csv`: Representative subset of the processed independent Batch 2 data from HC and ASD samples.

## MEA feature extraction and analysis

MEA features were extracted using the Neural Metrics Tool (Axion BioSystems). Python scripts were used for subsequent preprocessing, model training, evaluation, and prediction.

The XGBoost model was trained as a binary classifier, with the HC and ASD phenotypes labeled as 0 and 1, respectively. Batch 1 data were used for model training and validation, whereas independently generated Batch 2 data were used for testing. Further details of the analytical procedures are provided in the associated manuscript.

## Representative data

The CSV files contain de-identified subsets of the processed experimental MEA data. Their column structure and class labels are consistent with those used in the analysis, and the sample identifiers have been anonymized.

These subsets are provided to demonstrate the required input-data structure and analytical workflow. Because they include only part of the complete dataset, they are not intended to reproduce the performance metrics reported in the manuscript.

## Usage

The analytical workflow is implemented in `learn_and_predict.py`. Before running the script, install the required Python packages and place the representative training and test datasets in the input locations specified in the script. File paths may need to be adjusted for the local environment.

The trained model can be loaded from `xgb_model.json` to perform predictions without retraining.

## Data availability

The complete MEA dataset is not publicly available because it includes data generated using a proprietary compound library and is subject to institutional and commercial data-sharing restrictions. De-identified data that can be shared within these restrictions may be made available from the corresponding author upon reasonable request, subject to the necessary approvals and execution of an appropriate data-use agreement.

## Contact

For questions regarding the study or data availability, please contact the corresponding author:

Haruhisa Inoue  
Center for iPS Cell Research and Application (CiRA), Kyoto University  
Email: haruhisa@cira.kyoto-u.ac.jp
