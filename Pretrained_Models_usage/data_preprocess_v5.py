'''
Lrasmy@Zhilab  Jan 2021

# This script processes originally extracted data on a distributed platform
# This code version create data with multiple labels for both survival and Binary classification can create files for a predefined split of patients or can randomly split
# also it can build upon existing typoes dictionary or creates its own
# and builds pickled lists including a full list that includes all information for case and controls
## it outputs pickled list of the following shape
#[[pt1_id,label,[
#                  [[delta_time 0],[list of Medical codes in Visit0]],
#                  [[delta_time between V0 and V1],[list of Medical codes in Visit2]],
#                   ......]],
# [pt2_id,label,[[[delta_time 0],[list of Medical codes in Visit0 ]],[[delta_time between V0 and V1],[list of Medical codes in Visit2]],......]]]
#
# This version is for multiple outcomes prediction, therefore the label looks like ['mort_label','LOS','vent_label','time_to_intub','Readmission_label','plos_label']; please note the LOS is equivalent to time to event for in-hospital mortality event.
#
# Usage: feed this script with Case file and Control files each is just a three columns like pt_id | medical_code | visit_date and execute like:
#
# python data_preprocessing_v4.py <Case File> <Control File> <types dictionary if available,otherwise use 'NA' to build new one> <output Files Prefix> <path and prefix to pts file if available,otherwise use 'NA' to build new one>
# you can optionally activate <case_samplesize> <control_samplesize> based on your cohort definition
# This file will later split the data to training , validation and Test sets of ratio
# Output files include
# <output file>.pts: List of unique Patient ids. Created for validation and comparison purposes
# <output file>.types: Python dictionary that maps string diagnosis codes to integer diagnosis codes.
# Main output files for the baseline RNN models are <output file>.combined
'''

import sys
from optparse import OptionParser
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import random
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import glob
#import timeit ( for time tracking if required)

if __name__ == '__main__':

    dataFile= sys.argv[1]
    labelFile= sys.argv[2]
    typeFile= sys.argv[3]
    outFile = sys.argv[4]
    pts_file_pre = sys.argv[5]
    #cls_type= sys.argv[6]
    #samplesize_pts = int(sys.argv[7])
    parser = OptionParser()
    (options, args) = parser.parse_args()
 
   
    #_start = timeit.timeit()
    debug=False
    #np.random.seed(1)
    time_list = []
    dates_list =[]
    label_list = []
    pt_list = []


    ## loading Case
    print('loading data')
    ##### please uncomment glob lines for distributed data
    #all_files1 = glob.glob(dataFile + "/*.csv")
    #li1 = []
    #for filename in all_files1:
    #    df = pd.read_csv(filename)
    #    li1.append(df)
    #data_dat = pd.concat(li1).drop_duplicates()
    data_dat=pd.read_table(dataFile)
    data_dat.columns = ["Pt_id", "ICD", "Time"]     
    
    #data_dat=data_dat[~(data_dat["ICD"].str.startswith('SNOMED') |data_dat["ICD"].str.startswith('L'))] ### use if you need to exclude or include only certain type of codes
    
    print('loaded data for: ',data_dat["Pt_id"].nunique())

    print('loading labels')

    #all_files = glob.glob(labelFile + "/*.csv")
    #li = []
    #for filename in all_files:
    #    df = pd.read_csv(filename)
    #    li.append(df)
    #data_lbl_v1 = pd.concat(li).drop_duplicates()
    data_lbl_v1=pd.read_table(labelFile)
    data_lbl_v1.columns = ["Pt_id", "mort_label","LOS","vent_label","time_to_intub","Readmission_label","plos_label"] 
    data_lbl=pd.merge(data_dat["Pt_id"].drop_duplicates(),data_lbl_v1, how='inner').drop_duplicates()
    print('loaded labels for: ',data_lbl_v1["Pt_id"].nunique() , ' after primary cleaning ',data_lbl["Pt_id"].nunique())
    print('Mortality Case counts: ',data_lbl[data_lbl["mort_label"]==1]["Pt_id"].nunique())
    print('Intubation Case counts: ',data_lbl[data_lbl["vent_label"]==1]["Pt_id"].nunique())
    print('Intubation Case with tti >=1 : ',data_lbl[(data_lbl["vent_label"]==1)& (data_lbl["time_to_intub"]>=1)]["Pt_id"].nunique())
    print('LOS>7 : ',data_lbl[data_lbl["LOS"]>7]["Pt_id"].nunique())
    print('pLOS>7 : ',data_lbl[data_lbl["plos_label"]==1]["Pt_id"].nunique())
    print('Readmission case counts : ',data_lbl[data_lbl["Readmission_label"]==1]["Pt_id"].nunique())

    ### An example of sampling code: Control Sampling
    #print('pt sampling')       
    #data_sk=data_dat["Pt_id"]
    #data_sk=data_sk.drop_duplicates()
    #data_sk_samp=data_sk.sample(n=samplesize_pts) ## that is an input arg 7
    #data_dat=data_dat[data_dat["Pt_id"].isin(data_sk_samp.values.tolist())]
    #data_lbl=data_lbl[data_lbl["Pt_id"].isin(data_sk_samp.values.tolist())]



    ## loading the types
    #### If using pretrained models, you should start from the shared types file

    if typeFile=='NA': 
       types={"zero_pad":0}
       print('new types dictionary')
    else:
      with open(typeFile, 'rb') as t2:
             types=pickle.load(t2)
      print('types dictionary loaded')

    #end_time = timeit.timeit()
    #print ("consumed time for data loading",(_start -end_time)/1000.0 )
    
    full_list=[]
    index_date = {}
    time_list = []
    dates_list =[]
    label_list = []
    pt_list = []
    dur_list=[]
    newVisit_list = []
    count=0

    for Pt, group in data_dat.groupby('Pt_id'):
            data_i_c = []
            data_dt_c = []
            for Time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False): ### ascending=True normal order ascending=False reveresed order
                        data_i_c.append(np.array(subgroup['ICD']).tolist())             
                        data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
            if len(data_i_c) > 0:
                 # creating the duration in days between visits list, first visit marked with 0   (last in reversed order)     
                    v_dur_c=[]
            if len(data_dt_c)<=1:
                     v_dur_c=[0]
            else:
                     for jx in range (len(data_dt_c)):
                        if jx==0:
                             v_dur_c.append(jx)
                        else:
                            #xx = ((dt.strptime(data_dt_c[jx-1], '%d-%b-%y'))-(dt.strptime(data_dt_c[jx], '%d-%b-%y'))).days ## use if original data have time information or different date format
                            #xx = (data_dt_c[jx]- data_dt_c[jx-1]).days ### normal order
                            xx = (data_dt_c[jx-1] - data_dt_c[jx]).days ## reversed order                            
                            v_dur_c.append(xx)

            ### Diagnosis recoding
            newPatient_c = []
            for visit in data_i_c:
                      newVisit_c = []
                      for code in visit:
                                    if code in types: newVisit_c.append(types[code])
                                    else:                             
                                          types[code] = max(types.values())+1
                                          newVisit_c.append(types[code])
                      newPatient_c.append(newVisit_c)

            if len(data_i_c) > 0: ## only save non-empty entries
                  label_list.append(data_lbl.loc[data_lbl.Pt_id == Pt, ['mort_label','LOS','vent_label','time_to_intub','Readmission_label','plos_label']].values.squeeze().tolist())  #### LR ammended for multilabel
                  pt_list.append(Pt)
                  newVisit_list.append(newPatient_c)
                  dur_list.append(v_dur_c)

            count=count+1
            if count % 1000 == 0: print ('processed %d pts' % count)


    ### Creating the full pickled lists ### uncomment if you need to dump the all data before splitting

    pickle.dump(types, open(outFile+'.types', 'wb'), -1)

  
    ### Create the combined list for the Pytorch RNN
    fset=[]
    print ('Reparsing')
    for pt_idx in range(len(pt_list)):
                pt_sk= pt_list[pt_idx]
                pt_lbl= label_list[pt_idx]
                pt_vis= newVisit_list[pt_idx]
                pt_td= dur_list[pt_idx]
                d_gr=[]
                n_seq=[]
                d_a_v=[]
                for v in range(len(pt_vis)):
                        nv=[]
                        nv.append([pt_td[v]])
                        nv.append(pt_vis[v])                   
                        n_seq.append(nv)
                n_pt= [pt_sk,pt_lbl,n_seq]
                fset.append(n_pt)    
                
                          
    ### Random split to train ,test and validation sets, if needed 
    print ("Splitting")
    
    if pts_file_pre=='nosplit':
      nptfile = outFile +'.pts.all'
      pickle.dump(pt_list, open(nptfile, 'wb'),protocol=2)    
      train_set_full = [fset[i] for i in train_indices]
      filename_all=outFile+'.combined.all'    
      pickle.dump(fset, open(filename_all, 'wb'), -1)
    
    else:
      if pts_file_pre=='NA':
          print('random split')
          dataSize = len(pt_list)
          #np.random.seed(0)
          ind = np.random.permutation(dataSize)
          nTest = int(0.2 * dataSize)
          nValid = int(0.1 * dataSize)
          test_indices = ind[:nTest]
          valid_indices = ind[nTest:nTest+nValid]
          train_indices = ind[nTest+nValid:]
      else:
          print ('loading previous splits')
          pt_train=pickle.load(open(pts_file_pre+'.train', 'rb'))
          pt_valid=pickle.load(open(pts_file_pre+'.valid', 'rb'))
          pt_test=pickle.load(open(pts_file_pre+'.test', 'rb'))
          test_indices = np.intersect1d(pt_list, pt_test,assume_unique=True, return_indices=True)[1]
          valid_indices= np.intersect1d(pt_list, pt_valid,assume_unique=True, return_indices=True)[1]
          train_indices= np.intersect1d(pt_list, pt_train,assume_unique=True, return_indices=True)[1]
  
      for subset in ['train','valid','test']:
          if subset =='train':
              indices = train_indices
          elif subset =='valid':
              indices = valid_indices
          elif subset =='test':
              indices = test_indices
          else: 
              print ('error')
              break
          
          ### split the full combined set to the same as individual files
          #### only using Pts file for later, so dumping them 
  
          subset_p = [pt_list[i] for i in indices]
          nptfile = outFile +'.pts.'+subset
          pickle.dump(subset_p, open(nptfile, 'wb'),protocol=2)    
          subset_nfull = [fset[i] for i in indices]
          csfilename=outFile+'.combined.'+subset
          pickle.dump(subset_nfull, open(csfilename, 'wb'), -1)

