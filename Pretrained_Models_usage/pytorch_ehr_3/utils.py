# -*- coding: utf-8 -*-
# coding: utf-8
# -*- coding: utf-8 -*-
"""
This Class is mainly for the creation of the EHR patients' visits embedding
which is the key input for all the deep learning models in this Repo
@authors: Lrasmy , Jzhu @ DeguiZhi Lab - UTHealth SBMI
Last revised Feb 20 2020
"""
from __future__ import print_function, division
from io import open
#import string
#import re
import random
import math 
import time 
import os

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 
import sklearn.metrics as m
import matplotlib.pyplot as plt
import numpy as np

#Rename afterwards
#import .EHRDataloader as EHRDataloader
from .EHRDataloader import iter_batch2
from termcolor import colored



use_cuda = torch.cuda.is_available()


###### minor functions, plots and prints
#loss plot
def showPlot(points):
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.show()

 
#auc_plot 
def auc_plot(y_real, y_hat):
    fpr, tpr, _ = roc_curve(y_real,  y_hat)
    auc = roc_auc_score(y_real, y_hat)
    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.show();

        
#time Elapsed
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)    


#print to file function
def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()

##### ML & Eval functions
def extra_metrics(yreal,yhat):
    ap_score= m.average_precision_score(yreal,yhat)
    tn, fp, fn, tp = m.confusion_matrix(yreal, (np.array(yhat)>0.5)).ravel()
    class_rep_dic=m.classification_report(yreal, (np.array(yhat)>0.5), output_dict=True,digits=4)
    return ap_score,tn, fp, fn, tp,class_rep_dic

def ml_evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    pred_prob=model.predict_proba(test_features)
    auc_p=roc_auc_score(test_labels,pred_prob[:,1])
    ap_score= m.average_precision_score(test_labels,pred_prob[:,1])
    tn, fp, fn, tp = m.confusion_matrix(test_labels, predictions).ravel()
    class_rep_dic=m.classification_report(test_labels,predictions,digits=4, output_dict=True)
    print('Model Performance')
    print('AUC = {:0.2f}%.'.format(auc_p*100))
    print('Confusion Matrix tn, fp, fn, tp:',tn, fp, fn, tp )
    print('Classification Report :',class_rep_dic)
    return test_labels,pred_prob[:,1],auc_p,ap_score,tn, fp, fn, tp,class_rep_dic#test_labels,pred_prob[:,1]


###### major model training utilities
def trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = nn.BCELoss()): 
    model.train() ## LR added Jul 10, that is the right implementation
    model.zero_grad()

    output = model(sample,seq_l, mtd)
    if (((label_tensor.shape[-1]>1)&(criterion==nn.CrossEntropyLoss))) :
        mc=output.shape[-1]
        e,d=label_tensor.squeeze().T
        d_m=d
        d_m[d_m > mc-3]= mc-2
        d_m[e==0]= mc-1      
        if use_cuda:
            lnt_typ=torch.cuda.LongTensor
        else: 
            lnt_typ=torch.LongTensor
        criterion=nn.CrossEntropyLoss(reduction='sum')
        loss = criterion(output, d_m.view(-1).type(lnt_typ))
    else:
        loss = criterion(output.squeeze(), label_tensor.squeeze())  ##LR 9/2/21 added squeez for compatability with Pytorch1.7
        
    loss.backward()   
    optimizer.step()
    # print(loss.item())
    return output, loss.item()


#train with loaders

def trainbatches(mbs_list, model, optimizer,shuffle = True, loss_fn = nn.BCELoss()):#,we dont need this print print_every = 10, plot_every = 5): 
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    if shuffle: 
         # you can also shuffle batches using iter_batch2 method in EHRDataloader
        #  loader = iter_batch2(mbs_list, len(mbs_list))
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        output, loss = trainsample(sample, label_tensor, seq_l, mtd, model, optimizer, criterion = loss_fn)### LR amended Sep 30 2020 to make sure we can change the loss function for survival
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
    return current_loss, all_losses 


def trainbatches_multilabel(mbs_list, model, optimizer,shuffle = True, loss_fn = nn.BCELoss()):#,we dont need this print print_every = 10, plot_every = 5): 
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    if shuffle: 
         # you can also shuffle batches using iter_batch2 method in EHRDataloader
        #  loader = iter_batch2(mbs_list, len(mbs_list))
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        task_label_tensor=label_tensor[:,:,0] ### for mortality testing
        output, loss = trainsample(sample, task_label_tensor, seq_l, mtd, model, optimizer, criterion = loss_fn)### LR amended Sep 30 2020 to make sure we can change the loss function for survival
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
    return current_loss, all_losses 


def trainbatches_outcomes(mbs_list, model,task, optimizer,shuffle = True, loss_fn = nn.BCELoss()):#,we dont need this print print_every = 10, plot_every = 5): 
    current_loss = 0
    all_losses =[]
    plot_every = 5
    n_iter = 0 
    if shuffle: 
         # you can also shuffle batches using iter_batch2 method in EHRDataloader
        #  loader = iter_batch2(mbs_list, len(mbs_list))
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):
        sample, label_tensor, seq_l, mtd = batch
        if task=='mort':
            task_label_tensor=label_tensor[:,:,0] ### for mortality testing
        elif task=='vent':
            task_label_tensor=label_tensor[:,:,2] ### for mortality testing
        elif task=='readm':
            task_label_tensor=label_tensor[:,:,4] ### for mortality testing
        elif task=='plos':
            task_label_tensor=label_tensor[:,:,5] ### for mortality testing
        elif task=='mort_surv':
            task_label_tensor=label_tensor[:,:,0:2] ### for mortality testing
        elif task=='vent_surv':
            task_label_tensor=label_tensor[:,:,2:4] ### for mortality testing
        else: print (" task need to be one of ('mort','vent','readm','plos','mort_surv','vent_surv')")

        output, loss = trainsample(sample, task_label_tensor, seq_l, mtd, model, optimizer, criterion = loss_fn)### LR amended Sep 30 2020 to make sure we can change the loss function for survival
        current_loss += loss
        n_iter +=1
    
        if n_iter % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0    
    return current_loss, all_losses 


def calculate_auc(model, mbs_list, which_model = 'RNN', shuffle = True,mc=1): # batch_size= 128 not needed
    model.eval() ## LR added Jul 10, that is the right implementation
    y_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        output = model(sample, seq_l, mtd)
   
        
        if ((label_tensor.shape[-1]>1) & (mc > 1)): ### DeepHIT scenario
            e,d=label_tensor.squeeze().T
            d_m=d.cpu().data.numpy()
            d_m[d_m > mc-3]= mc-2
            d_m[e.cpu().data.numpy()==0]= mc-1
            y_real.extend(d_m)
            y_hat.extend(output.cpu().data.numpy())  
            
        elif (mc > 1):
            y_real.extend(label_tensor.cpu().data.view(-1).numpy())
            y_hat.extend(output.cpu().data.numpy())              
        else:
            y_real.extend(label_tensor.cpu().data.view(-1).numpy())
            y_hat.extend(output.cpu().data.view(-1).numpy())  
 
    if mc>1:    
        #print(np.array(y_hat).shape , np.array(y_real).shape, max(y_real) )
        auc = roc_auc_score(y_real, y_hat,labels=np.arange(mc),multi_class='ovo',average='macro')
    else:     
        auc = roc_auc_score(y_real, y_hat)
    
    return auc, y_real, y_hat 

def calculate_auc_multilabel(model, mbs_list, which_model = 'RNN', shuffle = True,mc=1): # batch_size= 128 not needed
    model.eval() ## LR added Jul 10, that is the right implementation
    y_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        task_label_tensor=label_tensor[:,:,0] ### for mortality testing

        output = model(sample, seq_l, mtd)
   
        
        if ((task_label_tensor.shape[-1]>1) & (mc > 1)): ### DeepHIT scenario
            e,d=task_label_tensor.squeeze().T
            d_m=d.cpu().data.numpy()
            d_m[d_m > mc-3]= mc-2
            d_m[e.cpu().data.numpy()==0]= mc-1
            y_real.extend(d_m)
            y_hat.extend(output.cpu().data.numpy())  
            
        elif (mc > 1):
            y_real.extend(task_label_tensor.cpu().data.view(-1).numpy())
            y_hat.extend(output.cpu().data.numpy())              
        else:
            y_real.extend(task_label_tensor.cpu().data.view(-1).numpy())
            y_hat.extend(output.cpu().data.view(-1).numpy())  
 
    if mc>1:    
        #print(np.array(y_hat).shape , np.array(y_real).shape, max(y_real) )
        auc = roc_auc_score(y_real, y_hat,labels=np.arange(mc),multi_class='ovo',average='macro')
    else:     
        auc = roc_auc_score(y_real, y_hat)
    
    return auc, y_real, y_hat 


def calculate_auc_outcomes(model, mbs_list, task, which_model = 'RNN', shuffle = True,mc=1): # batch_size= 128 not needed
    model.eval() ## LR added Jul 10, that is the right implementation
    y_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        
        if task=='mort':
            task_label_tensor=label_tensor[:,:,0] ### for mortality testing
        elif task=='vent':
            task_label_tensor=label_tensor[:,:,2] ### for intubation testing
        elif task=='readm':
            task_label_tensor=label_tensor[:,:,4] ### for readmision testing
        elif task=='plos':
            task_label_tensor=label_tensor[:,:,5] ### for prolonged LOS testing
        else: print (" task need to be one of ('mort','vent','readm','plos')")

        output = model(sample, seq_l, mtd)
   
        
        if ((task_label_tensor.shape[-1]>1) & (mc > 1)): ### DeepHIT scenario
            e,d=task_label_tensor.squeeze().T
            d_m=d.cpu().data.numpy()
            d_m[d_m > mc-3]= mc-2
            d_m[e.cpu().data.numpy()==0]= mc-1
            y_real.extend(d_m)
            y_hat.extend(output.cpu().data.numpy())  
            
        elif (mc > 1):
            y_real.extend(task_label_tensor.cpu().data.view(-1).numpy())
            y_hat.extend(output.cpu().data.numpy())              
        else:
            y_real.extend(task_label_tensor.cpu().data.view(-1).numpy())
            y_hat.extend(output.cpu().data.view(-1).numpy())  
 
    if mc>1:    
        #print(np.array(y_hat).shape , np.array(y_real).shape, max(y_real) )
        auc = roc_auc_score(y_real, y_hat,labels=np.arange(mc),multi_class='ovo',average='macro')
    else:     
        auc = roc_auc_score(y_real, y_hat)
    
    return auc, y_real, y_hat 


    
#define the final epochs running, use the different names

def epochs_run(epochs, train, valid, test, model, optimizer, shuffle = True, which_model = 'RNN', patience = 20, output_dir = '../models/', model_prefix = 'dhf.train', model_customed= ''):  
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    #header = 'BestValidAUC|TestAUC|atEpoch'
    #logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    #print2file(header, logFile)
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = trainbatches(mbs_list = train, model= model, optimizer = optimizer)

        train_time = timeSince(start)
        #epoch_loss.append(train_loss)
        avg_loss = np.mean(train_loss)
        valid_start = time.time()
        train_auc, _, _ = calculate_auc(model = model, mbs_list = train, which_model = which_model, shuffle = shuffle)
        valid_auc, _, _ = calculate_auc(model = model, mbs_list = valid, which_model = which_model, shuffle = shuffle)
        valid_time = timeSince(valid_start)
        print(colored('\n Epoch (%s): Train_auc (%s), Valid_auc (%s) ,Training Average_loss (%s), Train_time (%s), Eval_time (%s)'%(ep, train_auc, valid_auc , avg_loss,train_time, valid_time), 'green'))
        if valid_auc > bestValidAuc: 
              bestValidAuc = valid_auc
              bestValidEpoch = ep
              best_model= model 
              if test:      
                      testeval_start = time.time()
                      bestTestAuc, _, _ = calculate_auc(model = best_model, mbs_list = test, which_model = which_model, shuffle = shuffle)
                      print(colored('\n Test_AUC (%s) , Test_eval_time (%s) '%(bestTestAuc, timeSince(testeval_start)), 'yellow')) 
                      #print(best_model,model) ## to verify that the hyperparameters already impacting the model definition
                      #print(optimizer)
        if ep - bestValidEpoch > patience:
              break
    #if test:      
    #   bestTestAuc, _, _ = calculate_auc(model = best_model, mbs_list = test, which_model = which_model, shuffle = shuffle) ## LR code reorder Jul 10
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #save model & parameters
    torch.save(best_model, output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    torch.save(best_model.state_dict(), output_dir + model_prefix + model_customed + 'EHRmodel.st')
    '''
    #later you can do to load previously trained model:
    best_model= torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.pth')
    best_model.load_state_dict(torch.load(args.output_dir + model_prefix + model_customed + 'EHRmodel.st'))
    best_model.eval()
    '''
    #Record in the log file , modify below as you like, this is just as example
    header = 'BestValidAUC|TestAUC|atEpoch'
    logFile = output_dir + model_prefix + model_customed +'EHRmodel.log'
    print2file(header, logFile)
    pFile = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch)
    print2file(pFile, logFile) 
    if test:
        print(colored('BestValidAuc %f has a TestAuc of %f at epoch %d ' % (bestValidAuc, bestTestAuc, bestValidEpoch),'green'))
    else: 
        print(colored('BestValidAuc %f at epoch %d ' % (bestValidAuc,  bestValidEpoch),'green'))
        print('No Test Accuracy')
    print(colored('Details see ../models/%sEHRmodel.log' %(model_prefix + model_customed),'green'))


#### LR Sep 29 ## adding cox_ph_loss following the code available on https://github.com/havakv/pycox --pycox/models/loss.py#L425

def cox_ph_loss(log_h, label, eps=1e-7):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    events, durations = label.squeeze().T
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    #print('events after sort',events) ###verified the shape
    log_h = log_h[idx]
    #print('log_h',log_h) ###verified the shape
    return cox_ph_loss_sorted(log_h, events, eps)
    
def cox_ph_loss_sorted(log_h, events, eps = 1e-7):
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


    
def PartialLogLikelihood(logits, label, ties= 'noties'):
    '''
    fail_indicator: 1 if the sample fails, 0 if the sample is censored.
    logits: raw output from model 
    ties: 'noties' or 'efron' or 'breslow'
    '''
    
    fail_indicator, durations = label.squeeze().T
    logL = 0
    # pre-calculate cumsum
    cumsum_y_pred = torch.cumsum(logits, 0)
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, 0)
    if ties == 'noties':
        log_risk = torch.log(cumsum_hazard_ratio)
        likelihood = logits - log_risk
        # dimension for E: np.array -> [None, 1]
        uncensored_likelihood = likelihood * fail_indicator
        logL = -torch.sum(uncensored_likelihood)
    else:
        raise NotImplementedError()
    # negative average log-likelihood
    observations = torch.sum(fail_indicator, 0)
    return 1.0*logL / observations

from lifelines.utils import concordance_index

def c_index(risk_pred, e,y): #### not actually used
    ''' Performs calculating c-index
    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def calculate_cindex(model, mbs_list, which_model = 'RNN', shuffle = True): # batch_size= 128 not needed
    model.eval() 
    e_real =[]
    d_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        output = model(sample, seq_l, mtd)
        y_hat.extend(output.cpu().data.view(-1).numpy()*-1)  
        e, d = label_tensor.squeeze().T
        d_real.extend(d.cpu().data.view(-1).numpy())
        e_real.extend(e.cpu().data.view(-1).numpy())
       
    c_index = concordance_index(d_real, y_hat,e_real)
    return c_index, (d_real,e_real), y_hat 

def calculate_cindex_outcomes(model, mbs_list, task, which_model = 'RNN', shuffle = True): # batch_size= 128 not needed
    model.eval() 
    e_real =[]
    d_real =[]
    y_hat= []
    if shuffle: 
        random.shuffle(mbs_list)
    for i,batch in enumerate(mbs_list):

        sample, label_tensor, seq_l, mtd = batch
        if task=='mort_surv':
            task_label_tensor=label_tensor[:,:,0:2] ### for mortality testing
        elif task=='vent_surv':
            task_label_tensor=label_tensor[:,:,2:4] ### for mortality testing
        else: print (" task need to be one of ('mort_surv','vent_surv')")

        output = model(sample, seq_l, mtd)
        y_hat.extend(output.cpu().data.view(-1).numpy()*-1)  
        e, d = task_label_tensor.squeeze().T
        d_real.extend(d.cpu().data.view(-1).numpy())
        e_real.extend(e.cpu().data.view(-1).numpy())
       
    c_index = concordance_index(d_real, y_hat,e_real)
    return c_index, (d_real,e_real), y_hat 


####Plots
def plot_roc_curve(label,score):
    fpr, tpr, ths = m.roc_curve(label, score) ### If I round it gives me an AUC of 64%
    roc_auc = m.auc(fpr, tpr)
    ### add aditional measures
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_roc_curve_combined_m(title,list_of_tuples):

    plt.figure()
    for tup in list_of_tuples:
        model_name, true_label, pred_score=tup
        fpr, tpr, ths = m.roc_curve(true_label, pred_score) ### If I round it gives me an AUC of 64%
        roc_auc = m.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3, label='%s (AUC = %0.1f%%)'%(model_name,roc_auc*100))

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title(title,fontdict={'fontsize':15,'fontweight' :800} )
    plt.legend(loc="lower right",fontsize=14)
    plt.show()

#### Calibration Plot
from sklearn.calibration import calibration_curve
def plot_calibration_curve(name, fig_index,y_test, probs):
    """Plot calibration curve for est w/o and with calibration. """

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    frac_of_pos, mean_pred_value = calibration_curve(y_test, probs, n_bins=10)

    ax1.plot(mean_pred_value, frac_of_pos, "s-", label=f'{name}')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'Calibration plot ({name})')
    
    ax2.hist(probs, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
