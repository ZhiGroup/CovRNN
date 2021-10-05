

# ![image](https://user-images.githubusercontent.com/25290490/135663245-4bd357a3-52d8-4b65-a1b1-74012385ecc8.png) CovRNN: A collection of Recurrent Neural Network models for the prediction of COVID-19 patients outcomes on admission based on their electronic health records (EHR) data
This repository provides the code for training and fine-tuning CovRNN, a collection of Recurrent Neural Network models for the prediction of COVID-19 patients outcomes on admission based on their electronic health records (EHR) data on admission, without the need for specific feature selection or missing data imputation

## Overview
CovRNN is designed to predict three outcomes: in-hospital mortality, need for mechanical ventilation, and long length of stay (LOS >7 days). Predictions are made for time-to-event risk scores (survival prediction) and all-time risk scores (binary prediction). Our models were trained and validated using heterogeneous and de-identified data of 247,960 COVID-19 patients from 87 healthcare systems, derived from the Cerner® Real-World Dataset (CRWD) and 36,140 de-identified patients’ data derived from the Optum® de-identified COVID-19 Electronic Health Record v. 1015 dataset (2007–2020).
For further details, Please refer to our paper [CovRNN—A recurrent neural network model for predicting outcomes of COVID-19 patients: model development and validation using EHR data](https://www.medrxiv.org/content/10.1101/2021.09.27.21264121v1).

![image](https://user-images.githubusercontent.com/25290490/135668153-af3dec4f-1147-4fc1-aa06-f47fbb131484.png)

We showed that deep learning-based models can achieve state-of-the-art prediction accuracy while consuming the structured EHR categorical data in their standard raw format without the need for extensive feature engineering, which implies that the trained models can be easily validated on new data sources. CovRNN was validated across datasets from different sources, indicating it's transferability. Our framework can be further applied to train and evaluate predictive models for different types of clinical events. 

![image](https://user-images.githubusercontent.com/25290490/135629163-e11d8e9f-d88c-4ac6-8b84-9c993ecd25ed.png)


In this Repository, we are sharing:

(i) The pretrained CovRNN trained on more than 170,000 COVID-19 patients extracted from the CRWD, so you can fine-tune our CovRNN pre-trained model on a sample of your local data, and use it.

(ii) Our comprehensive model development framework to train a new predictive model using your own data.

## Results

![image](https://user-images.githubusercontent.com/25290490/135732094-a6d48662-f617-4ffa-bb02-73ee4d1e61f8.png)

![image](https://user-images.githubusercontent.com/25290490/135732106-cc269e56-93c3-4222-ab8c-87d0f8f5acd0.png)

## Folder structure
#### [Pretrained Model Usage] (https://github.com/ZhiGroup/CovRNN/tree/main/Pretrained_Models_usage) is described in this folder, including model fine-tuning
This folder also includes the CRWD pretrained models and their state dictionaries


## Citation

Rasmy L, Nigo M, Kannadath BS, Xie Z, Mao B, Patel K, Zhou Y, Zhang W, Ross AM, Xu H, Zhi D. CovRNN-A recurrent neural network model for predicting outcomes of COVID-19 patients: model development and validation using EHR data. medRxiv. 2021 Sep 29


