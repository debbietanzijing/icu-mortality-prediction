# icu-mortality-prediction
Predicting probability of death in an ICU 

## Overview 
The project is predicting the probability of death of a patient entering the Intensive Care Unit (ICU) with machine learning models. By having an accurate predictive system, it allows for identification of high risk patients and informed clinical decision making- delivering more effective and data- informed care. 

## Objective
To predict the probability of death for ICU patients using retrospective clinical data and machine learning models

## Dataset 
The dataset comes from the Medical Information Mart for Intensive Care III (MIMIC-III) project. It is an open source database comprising of deidentified health data associated with over forty thousand patients living in critical care units of Beth Israel Deaconess Medical Center between 2001 and 2012 

The dataset includes key vital signs, diagnosis and general characteristics (age, gender) and clincial outcome (mortality) of each patient. More deails can be found under _mimic_patient_metadata.csv_

## Methodology
* Initial data exploration
* Feature engineering (age, one- hot encoding (more to be added))
* Model selection (logistic regression, XGBoost (more to be added))
* Evaluation metrics (AUC ROC)

  ## Results

  ## Installation/ usage
  (how can someone run this code)

  ## Future work
  (ideas for improving/ extending the project?) 

# Evaluation 
The main evaluation metric used will be ROC AUC. This is chosen as a common metric used in binary classification models. 
