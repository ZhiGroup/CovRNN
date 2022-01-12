## Data Extraction from Cerner RWD HealtheDataLab and pre-processing

##### This section include code used to  describe data extraction from CRWD HealtheDataLab

### 1. Cohort definition 

kindly refer to the [Cohort Definition notebook](Data%20Extraction/Cohort_Definition_CRWD.ipynb) for further details


### 2. Input Data Extraction

kindly refer to the [Cohort Definition notebook](Data%20Extraction/Data_Extraction_CRWD.ipynb) for further details

### 3. Data Preprocessing

Data extracted from the previous data extraction steps will mainly preprocessed using the 

  python data_preprocessing_v4.py LR_inhospDec_dmlpd_all_dat.csv LR_inhospDec_dmlpd_all_labelv1.csv NA pdata/lr_inhosp_outcome_pred_v1 NA
 
The [Preprocessing example notebook] illustrate the preprocessing code calling example and how we can use and manipulate the output to later use for model training or finetuning.
