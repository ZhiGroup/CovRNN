### Tools and Packages
##Basics
import pandas as pd
import numpy as np
import sys, random
import math
try:
    import cPickle as pickle
except:
    import pickle
import string
import re
import os
import time
from tqdm import tqdm

## ML and Stats 
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
import sklearn.linear_model  as lm
import lifelines#.estimation import KaplanMeierFitter
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import export_graphviz
import statsmodels.formula.api as sm
import patsy
from scipy import stats
from termcolor import colored


## Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
%matplotlib inline
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.graph_objs import *
from IPython.display import HTML

## DL Framework
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

###GPU enabling and device allocation
use_cuda = torch.cuda.is_available()
#torch.cuda.set_device(2)


from importlib import reload

### import pytorch ehr files
#import sys
#sys.path.insert(0, '../ehr_pytorch')
import pytorch_ehr_3.models as model 
from pytorch_ehr_3.EHRDataloader import EHRdataloader
from pytorch_ehr_3.EHRDataloader import EHRdataFromLoadedPickles as EHRDataset
import pytorch_ehr_3.utils as ut 
from pytorch_ehr_3.EHREmb import EHREmbeddings


############


def load_mbs_var(sets_list,packpadmode,multiLbl=True,bs=128):
    mb_lists=[]
    for dset in sets_list:
        print (' creating the list of minibatches')
        dataset = EHRDataset(dset,sort= True, model='RNN')
        data_mbs = list(tqdm(EHRdataloader(dataset, batch_size = bs, packPadMode = packpadmode,multilbl=multiLbl)))
        mb_lists.append(data_mbs)
    return mb_lists


def run_pretrained_eval(tests_mbs,pretrained_modelname,packpadmode,task):
    loaded_model= torch.load(pretrained_modelname+'.pth')
    loaded_model.load_state_dict(torch.load(pretrained_modelname+'.st'))

    if use_cuda:loaded_model.cuda()
    loaded_model.eval()
    TestAucs=[]
    #y_real_f=[]
    #y_hat_f=[]
    for i,test_mbs in enumerate(tests_mbs):
        eval_start = time.time()
        TestAuc, y_real, y_hat = ut.calculate_auc_outcomes(loaded_model, test_mbs, task=task, which_model = 'RNN')
        eval_time = ut.timeSince(eval_start)
        print (i," Test_auc : " , str(TestAuc) ,' ,Eval Time :' ,str(eval_time))
        print(colored('\n Confusion matrix'), 'yellow')
        print(m.confusion_matrix(y_real, (np.array(y_hat)>0.5)))
        print('\n Test set Classification Report:', m.classification_report(y_real, (np.array(y_hat)>0.5)))
        TestAucs.append(TestAuc)
        #y_real_f.append(y_real)
        #y_hat_f.append(y_hat)

    return TestAucs#,y_real_f, y_hat_f

def run_pretrained_eval_surv_model(tests_mbs,loaded_model,packpadmode,task):
    loaded_model.eval()
    TestAucs=[]
    for i,test_mbs in enumerate(tests_mbs):
        eval_start = time.time()
        TestAuc, y_real, y_hat =  ut.calculate_cindex_outcomes(loaded_model, test_mbs, task=task, which_model = 'RNN')
        eval_time = ut.timeSince(eval_start)
        print (i," Test_cindex : " , str(TestAuc) ,' ,Eval Time :' ,str(eval_time))
        TestAucs.append(TestAuc)
    return TestAucs

def run_pretrained_eval_model(tests_mbs,loaded_model,packpadmode,task):
    loaded_model.eval()
    TestAucs=[]
    for i,test_mbs in enumerate(tests_mbs):
        eval_start = time.time()
        TestAuc, y_real, y_hat = ut.calculate_auc_outcomes(loaded_model, test_mbs, task=task, which_model = 'RNN')
        eval_time = ut.timeSince(eval_start)
        print (i," Test_auc : " , str(TestAuc) ,' ,Eval Time :' ,str(eval_time))
        print(colored('\n Confusion matrix'), 'yellow')
        print(m.confusion_matrix(y_real, (np.array(y_hat)>0.5)))
        print('\n Test set Classification Report:', m.classification_report(y_real, (np.array(y_hat)>0.5)))
        TestAucs.append(TestAuc)

    return TestAucs


### Fine tune or Train New RNN, based on the loaded model

def run_dl_model(ehr_model,train_mbs,valid_mbs,tests_mbs,bmodel_pth,bmodel_st,wmodel,packpadmode,task):
    ## Data Loading
    if task: multiLbl=True
    ## Data Loading

    ##Hyperparameters -- Fixed for testing purpose
    epochs = 100
    l2 = 0.0001
    lr = 0.05
    eps = 1e-4
    w_model= wmodel
    optimizer = optim.Adagrad(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)   
    

    ##Training epochs
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep =[]
    valid_auc_allep =[]
    test_auc_allep=[] 
    weights=torch.tensor([8])
    if use_cuda: weights=weights.to(device='cuda')
    print (weights)
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = ut.trainbatches_outcomes(train_mbs, model= ehr_model, task=task, optimizer = optimizer,loss_fn = nn.BCELoss(weight=weights))
        avg_loss = np.mean(train_loss)
        train_time = ut.timeSince(start)
        eval_start = time.time()
        Train_auc, y_real, y_hat  = ut.calculate_auc_outcomes(ehr_model, train_mbs, task=task, which_model = w_model)
        valid_auc, y_real, y_hat  = ut.calculate_auc_outcomes(ehr_model, valid_mbs, task=task, which_model = w_model)
        print ("Epoch: " ,str(ep) ," Train_auc :" , str(Train_auc) , " , Valid_auc : " ,str(valid_auc) ," Avg Loss: " ,str(avg_loss), ' , Train Time :' , str(train_time) )
        train_auc_allep.append(Train_auc)
        valid_auc_allep.append(valid_auc)

        if valid_auc > bestValidAuc: 
            TestAucs=[]
            y_reals=[]
            y_hats=[]
            for test_mbs1 in tests_mbs:
                TestAuc, y_real, y_hat = ut.calculate_auc_outcomes(ehr_model, test_mbs1, task=task, which_model = w_model)
                TestAucs.append(TestAuc)
                y_reals.append(y_real)
                y_hats.append(y_hat)
            test_auc_allep.append(TestAucs)
            print(" & Test_auc s : " , TestAucs  )
            eval_time = ut.timeSince(eval_start)
            print('Eval Time :' ,str(eval_time))

            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAucs
            print(colored('\n Confusion matrix'), 'yellow')
            for y_real,y_hat in zip(y_reals,y_hats):
                print(m.confusion_matrix(y_real, (np.array(y_hat)>0.5)))
                print('\n Classification Report:', m.classification_report(y_real, (np.array(y_hat)>0.5)))
            y_real_f=y_reals
            y_hat_f=y_hats
      
            ###uncomment the below lines to save the best model parameters
            best_model = ehr_model
            torch.save(best_model, bmodel_pth)
            torch.save(best_model.state_dict(), bmodel_st)
        if ep - bestValidEpoch >10: break
    print( 'bestValidAuc %f at epoch %d ' % (bestValidAuc,  bestValidEpoch))
    print( 'Test AUCs are ' , bestTestAuc )
    return train_auc_allep,valid_auc_allep,test_auc_allep,y_real_f, y_hat_f


def run_dl_model_surv(ehr_model,train_mbs,valid_mbs,tests_mbs,bmodel_pth,bmodel_st,wmodel,packpadmode,task):
    ## Data Loading
    if task: multiLbl=True
    ## Data Loading

    ##Hyperparameters -- Fixed for testing purpose
    epochs = 100
    l2 = 0.0004
    lr = 0.084
    eps = 0.0008
    w_model= wmodel
    optimizer = optim.Adagrad(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)   
    

    ##Training epochs
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep =[]
    valid_auc_allep =[]
    test_auc_allep=[] 
    #weights=torch.tensor([8])
    #if use_cuda: weights=weights.to(device='cuda')
    #print (weights)
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = ut.trainbatches_outcomes(train_mbs, model= ehr_model, task=task, optimizer = optimizer,loss_fn = ut.cox_ph_loss)
        avg_loss = np.mean(train_loss)
        train_time = ut.timeSince(start)
        eval_start = time.time()
        Train_auc, y_real, y_hat  = ut.calculate_cindex_outcomes(ehr_model, train_mbs, task=task,  which_model = w_model)
        valid_auc, y_real, y_hat  = ut.calculate_cindex_outcomes(ehr_model, valid_mbs, task=task,  which_model = w_model)
        print ("Epoch: " ,str(ep) ," Train_auc :" , str(Train_auc) , " , Valid_auc : " ,str(valid_auc) ," Avg Loss: " ,str(avg_loss), ' , Train Time :' , str(train_time) )
        train_auc_allep.append(Train_auc)
        valid_auc_allep.append(valid_auc)

        if valid_auc > bestValidAuc: 
            TestAucs=[]
            y_reals=[]
            y_hats=[]
            for test_mbs1 in tests_mbs:
                TestAuc, y_real, y_hat = ut.calculate_cindex_outcomes(ehr_model, test_mbs1, task=task, which_model = w_model)
                TestAucs.append(TestAuc)
                y_reals.append(y_real)
                y_hats.append(y_hat)
            test_auc_allep.append(TestAucs)
            print(" & Test_auc s : " , TestAucs  )
            eval_time = ut.timeSince(eval_start)
            print('Eval Time :' ,str(eval_time))

            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAucs
            #print(colored('\n Confusion matrix'), 'yellow')
            #for y_real,y_hat in zip(y_reals,y_hats):
                #print(m.confusion_matrix(y_real, (np.array(y_hat)>0.5)))
                #print('\n Classification Report:', m.classification_report(y_real, (np.array(y_hat)>0.5)))
            y_real_f=y_reals
            y_hat_f=y_hats
      
            ###uncomment the below lines to save the best model parameters
            best_model = ehr_model
            torch.save(best_model, bmodel_pth)
            torch.save(best_model.state_dict(), bmodel_st)
        if ep - bestValidEpoch >10: break
    print( 'bestValidAuc %f at epoch %d ' % (bestValidAuc,  bestValidEpoch))
    print( 'Test AUCs are ' , bestTestAuc )
    return train_auc_allep,valid_auc_allep,test_auc_allep,y_real_f, y_hat_f

def pt_predictions(test_set,mort_model,mort_surv_model,vent_model,vent_surv_model,plos_model):
    with torch.no_grad():
        pt_preds=[]
        for pt in test_set:
            #print(pt)
            pt_id=pt[0]
            pt_ds = EHRDataset([pt],sort= True, model='RNN')
            #print(pt_ds)
            pt_m = list(EHRdataloader(pt_ds, batch_size = 1, packPadMode = True,multilbl=True))
            #print(len(pt_m[0]))
            x1, label,seq_len,time_diff = pt_m[0]
            if use_cuda:
                label=label.cpu().squeeze().numpy()          
                mort_score = mort_model(x1,seq_len,time_diff).cpu().numpy()
                mort_surv_score = mort_surv_model(x1,seq_len,time_diff).cpu().numpy()
                vent_score = vent_model(x1,seq_len,time_diff).cpu().numpy()
                vent_surv_score = vent_surv_model(x1,seq_len,time_diff).cpu().numpy()
                plos_score = plos_model(x1,seq_len,time_diff).cpu().numpy()
            else:  
                label=label.squeeze().numpy()
                mort_score = mort_model(x1,seq_len,time_diff).numpy()
                mort_surv_score = mort_surv_model(x1,seq_len,time_diff).numpy()
                vent_score = vent_model(x1,seq_len,time_diff).numpy()
                vent_surv_score = vent_surv_model(x1,seq_len,time_diff).numpy()
                plos_score = plos_model(x1,seq_len,time_diff).numpy()
            pt_preds.append([pt_id,label[0],label[1],mort_score,mort_surv_score,label[2],label[3],vent_score,vent_surv_score,label[5],plos_score])
    
    pt_preds_df= pd.DataFrame(pt_preds)
    pt_preds_df.columns=['pt','mort_label','mort_tte','mort_prob','mort_logHF','vent_label','vent_tte','vent_prob','vent_logHF','plos_label','plos_prob']
    return pt_preds_df
