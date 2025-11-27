# icu-mortality-prediction
Predicting probability of death in an ICU 

# 1. Introduction 
## 1.1 Background 

Intensive Care Unit (ICU) admissions have increased considerably in the last decade. Early prediction of ICU mortality is crucial, as the critically ill tend to deteriorate rapidly and often without obvious early signs. Medeiros's study also identified patients that remain poorly characterized tends die within the first 24 hours following ICU admission. Accurately identifying high risk patients within the first ICU admission will enable: 
* Proactive decision making 
* More efficient resource allocation
* Reduction of preventable mortality

Machine learning has now also been increasingly used in critica care prediction tasks. 
## 1.2 Objective 

* Preprocess the dataset to build several predictive machine learning (with binary classification outcomes)
* Comparing accuracy of traditiional statistical models (Logistic Regresson) with more advanced machine learning models (Randon Forest, XGBoost)
* Evaluate model performance 

## 1.3 Scope and Limitation 

The dataset will be restricted to Medical Information Mart for Intensive Care III (MIMIC-III) derived cohort. It is an open source database comprising of deidentified health data associated with over forty thousand patients living in critical care units of Beth Israel Deaconess Medical Center between 2001 and 2012. The dataset includes key vital signs, diagnosis and general characteristics (age, gender) and clincial outcome (mortality) of each patient. More deails can be found under _mimic_patient_metadata.csv_

# 2. Data 
## 2.1 Data Source 

The dataset comprised of a combination of various tables found in the [MIMIC website](https://mimic.mit.edu/docs/iii/tables/) (There is currently a more updated version available). The database includes information on demographics, vital signs measurements, test results, ICD codes, mortality status amongst others. 

## 2.2 Outcome Variable 
* HOSPITAL_EXPIRE_FLAG (0 = survived, 1 = died)
  
## 2.3 Predictor variables: 
Identifiers: `subject_id`, `hadm_id`, `icustay_id`
Physiology: `HeartRate`, `SysBP`, `DiasBP`, `RespRate`, `TempC`, `SPO2`, `Glucose` 
Demographics: `GENDER`, `DOB`, `INSURNACE`, `RELIGION`, `MARITAL_STATUS`
Diagnosis: `Diagnosis`, `ICD9_diagnosis` 
Others: `ADMITTIME`, `DIFF`, `ADMISSION_TYPE`, `FIRST_CAREUNIT`

## 2.4 Train- Test Dataset Split 
Test- 5,221 observations
Train- 20, 855 observations

## 3. Initial Data Exploration 

### 3.1 Data Cleaning 
* There is a class imbalance in the dataset (11% mortality rate)
* There are no missing values in the dataset

### 3.2 Handling categorical variables 

A general rule of thumb (10) has been set to determine if a variable was at risk for high sparsity. High cardinality variables creates very sparse, huge matrices, and will result in overfitting, along with making modelling unstable. All categorical variables with the exception of `Diagnosis` have <10 cardinality.

Two methods were explored to manage `Diagnosis`: 
1. Target encoding
* Concepts: Laplace smoothing, Bayesian prior

2. Regex
Keyword matching pattern was used to group diagnosis into broader categories (e.g. 'resp|pneumon|bronch under 'Respiratory' conditions). Conditions with a frequency count of >20 was left as it is, and unmatched conditions were grouped under 'Rare diseases'. Overall, I achieved a 75% reduction in cardinality- but still had significant number of 130 diagnosises uncategorised. Therefore, target methodology will be used instead.  



* Feature engineering (age, one- hot encoding (more to be added))


* Model selection (logistic regression, XGBoost (more to be added))
* Evaluation metrics (AUC ROC)

## Results/ Evaluation
The main evaluation metric used will be ROC AUC. This is chosen as a common metric used in binary classification models. 

## Future work
- ideas for improving/ extending the project (more to be added) 
