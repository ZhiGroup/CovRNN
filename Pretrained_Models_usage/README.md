

# This is a Beta version
## Under Testing 
after code cleaning and re-organization

### This section is describing how to use the pretrained model to get predictions on local test sets, with and without finetuning
However, we highly recommend fine-tuning the models on local data, for better performance 

#### Files:

1.[Get_predictions_withoutFinetunning.ipynb](Get_predictions_withoutFinetunning.ipynb) : this notebook shows how to  get predictions directly from the pretrained models without fine-tuning 

2.[Get_predictions_withFinetunning.ipynb](Get_predictions_withFinetunning.ipynb) : this notebook shows how to finetune the models  on 70% of your data and get predictions from the fine-tuned models (RECOMMENDED)

3.[data_preprocess_v5.py](data_preprocess_v5.py) : Data preprocessing code converting the extracted data and labels files into pickled lists consumed by pytorch_ehr

  
    python data_preprocessing_v5.py <Data File> <labels File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <path and prefix to pts file if available,otherwise use 'NA' for random slit into 7:1:2, or 'nosplit' for avoid splitting>
 
  - The data file, is a tab delimited file that has the patient data as :
  
    patient_identifier  event_code  encounter_date
  
  where patient_identifier is any number that can map the data to the labels records, please avoid using any PHI data
  , the event is any diagnosis, medication, preocedure, lab results... etc
  and the encounter date is either the admission or the discharge date (preferred) for the encounter where that event occurs
  
  - The label file, is also a tab delimited file that has the label for different outcomes like:
  patient_identifier  mortality_label Length_of_stay ventilation_label  time_to_intubation  Readmission_label plos_label
  
  - The types file used during CRWD pretraining [lr_inhosp_outcome_pred_v1.types](CRWD_Pretrained_Models/lr_inhosp_outcome_pred_v1.types) is available under [CRWD_Pretrained_Models](CRWD_Pretrained_Models) folder
  
  - After the preprocessing the output will look like:
    [[patient_1, ['mort_label','LOS','vent_label','time_to_intub','Readmission_label','plos_label'],
                 [[[delta_time 0],[list of Medical codes in Visit0]],
                  [[delta_time between V0 and V1],[list of Medical codes in Visit2]],
                   ...... ]],
      [patient_2, ['mort_label','LOS','vent_label','time_to_intub','Readmission_label','plos_label'],
                  [[[delta_time 0],[list of Medical codes in Visit0 ]],
                    [[delta_time between V0 and V1],[list of Medical codes in Visit2]]
                    ,......]] ]

  More details is included in the file header and the above mentioned notebooks include examples on how to use it.
  
  Other files and folders are including the pretrained models and modeling and utils code, adapted from our [pytorch_ehr](https://github.com/ZhiGroup/pytorch_ehr) repo. 
