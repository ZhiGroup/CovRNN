

# ![image](https://user-images.githubusercontent.com/25290490/135663245-4bd357a3-52d8-4b65-a1b1-74012385ecc8.png) CovRNN: A collection of Recurrent Neural Network models for the prediction of COVID-19 patients outcomes on admission based on their electronic health records (EHR) data

CovRNN, recurrent neural network (RNN)-based models to predict COVID-19 patients’ outcomes, using their available electronic health record (EHR) data on admission, without the need for specific feature selection or missing data imputation. CovRNN is designed to predict three outcomes: in-hospital mortality, need for mechanical ventilation, and long length of stay (LOS >7 days). Predictions are made for time-to-event risk scores (survival prediction) and all-time risk scores (binary prediction). Our models were trained and validated using heterogeneous and de-identified data of 247,960 COVID-19 patients from 87 healthcare systems, derived from the Cerner® Real-World Dataset (CRWD) and 36,140 de-identified patients’ data derived from the Optum® de-identified COVID-19 Electronic Health Record v. 1015 dataset (2007–2020).

![image](https://user-images.githubusercontent.com/25290490/135629163-e11d8e9f-d88c-4ac6-8b84-9c993ecd25ed.png)

We showed that deep learning-based models can achieve state-of-the-art prediction accuracy while consuming the structured EHR categorical data in their standard raw format without the need for extensive feature engineering, which implies that the trained models can be easily validated on new data sources. CovRNN was validated across datasets from different sources, indicating it's transferability. Our framework can be further applied to train and evaluate predictive models for different types of clinical events. 

In this Repository, we are sharing:

(i) The pretrained CovRNN trained on more than 170,000 COVID-19 patients extracted from the CRWD, so you can fine-tune our CovRNN pre-trained model on a sample of your local data, and use it.

(ii) Our comprehensive model development framework to train a new predictive model using your own data.

