## Data Extraction from Cerner RWD HealtheDataLab and pre-processing

##### This section include code used to  describe data extraction from CRWD HealtheDataLab

### 1. Cohort definition 

kindly refer to the [Cohort Definition notebook](Data%20Extraction/Cohort_Definition_CRWD.ipynb) for further details


### 2. Input Data Extraction

kindly refer to the [Cohort Definition notebook](Data%20Extraction/Data_Extraction_CRWD.ipynb) for further details

### 3. Data Preprocessing

Data extracted from the previous data extraction steps will mainly preprocessed using the 

  python data_preprocessing_v4.py <Case File> <Control File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <path and prefix to pts file if available,otherwise use 'NA' to build new one>
 
The [Preprocessing example notebook](Preprocessing_example_CRWD.ipynb) illustrate the preprocessing code calling example and how we can use and manipulate the output to later use for model training or finetuning.
