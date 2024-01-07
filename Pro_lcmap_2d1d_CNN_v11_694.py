#**************************************************************************************************************************************************************************************************
# v11_691 (used in the paper) on Dec 27, 2023
# v10_0 fix the use the right training tricks, i.e., (i) single schedule, (ii) DOY norm by mean and std, and (iii) logits = True should go without softmax 
    ## v10_01 fix the prediction adding softmax ! This is the version ! 
    ## v10_02 use 60 epochs rather than 70 (not as good as v10_01) 
    
# v10_1 same as 10_0 but consider pre-training 
    # v10_11 fix a bug with pre-training per epoch steps and fix v10_01 prediction softmax 
    # v10_12 on 20230316 fix epoch to be 20 and fix later epochs to be changable 
    # v10_13 to increase the size of the model: units = 256; head = 8; layern = 4 (not run yet) 
# all wrong abo v10_1* see v10_2 bug 

# v10_2 fix a mean, std norm bug when using proportion=1.0 
    # and fix a huge bug when random mask for training - only mask the reflectance not mask x,y not doy 
    # use subset 
    # v10_21 using 30e and 2*learning rate  
    # v10_22 using 30e and 2*learning rate and reduce 10 epochs in fine-tuning  (not good)
    # v10_23 same as v10_21 but only last layer is trainable 
    # v10_24 same as v10_21 but only last layer is trainable with random initialization 
    # v10_25 same as v10_21 but only first embedding layer is NOT trainable (!) 
    # v10_26 same as v10_21 but only first embedding layer and first Transformer block is NOT trainable (!) 
    # v10_27 same as v10_21 but only first embedding layer and first 2 Transformer block is NOT trainable (!) 
    # v10_28 same as v10_21 but only first embedding layer and all the 3 Transformer block is NOT trainable - same as v10_23 

# v10_30 same as v10_22 but use full 

# v10_40 same as v10_28 but use: units = 128; head = 4; layern = 4 (kept block=3) 
    # v10_41 kept block=3 
    # v10_42 kept block=2 (not good, so please use last block blank) 

# v10_50 same as v10_28 but use: units = 256; head = 8; layern = 4 (kept block=3) 
    # v10_51 kept block=4; # v10_52 kept block=3 (20e best); # v10_53 kept block=2
    # v10_54 kept block=3 and use 10-3 L2 (not good);     
    # can not leave last layer out general not good as 
    # v10_55 kept block=4 and use one more layer for classification 
    # v10_56 kept block=4 and use one more layer for classification with drop/relu (v10.90)

# v10_60 same as v10_40 but use: units = 128; head = 4; layern = 8 (kept block=7) 
    # v10_61 kept block=6 (no good) 
    # v10_62 1o 73 no good
        # v10_62 kept block=8 and do not use dropout and L2 in last layer (not good)
        # v10_63 kept block=8 and do not use dropout and L2 in last layer and a 1dense head to last layer (not good) 
        # v10_64 kept block=8 and a 1dense head to last layer (worse than v10_63) 
        # v10_65 kept block=8 and do not use dropout and L2 in last layer and a 1dense head to last layer and 10 time rates
        # v10_66 kept block=8 and do not use dropout and L2 in last layer and a 1dense head to last layer and 10 time rates fix dense bug (OK)
        # v10_67 kept block=8 and do not use dropout and L2 in last layer and a 3dense head to last layer and 10 time rates
        # v10_68 kept block=8 and do not use dropout and L2 in last layer and a 3dense head to last layer and 10 time rates and use L2
        # v10_69 kept block=8 and do not use dropout and L2 in last layer and a 3dense head to last layer and 10 time rates and use L2 and dropout (needed !) 
        # v10_70 kept block=8 and do not use dropout and L2 in last layer and a 3dense head to last layer and 10 time rates and use L2 and dropout (needed !) + 100 learning rate
        # v10_71 kept block=8 and do not use dropout and L2 in last layer and a 6dense head to last layer and 10 time rates and use L2 and dropout (needed !) + 10 learning rate
        # v10_72 kept block=8 and do not use dropout and L2 in last layer and a 6dense head to last layer and 10 time rates and use L2 and dropout (needed !) + same learning rate
        # v10_73 kept block=8 and do not use dropout and L2 in last layer and a 10dense head to last layer and 10 time rates and use L2 and dropout (needed !) + same learning rate
    # v10_74 kept block=7 (same as 10_60) 
    # v10_75 kept block=7 (same as 10_60 but all start from all data pretraining) not good
    # v10_76 kept block=8 but add one transfomer block for class model 
    # v10_77 kept block=8 but add one transfomer block for class model 
    # v10_78 kept block=8 but add one transfomer block for class model + add one linear 
    # v10_79 kept block=8 but add one transfomer block for class model + add one linear/dropout 
    # v10_90 kept block=8 but add one transfomer block for class model + add one linear/dropout/relu !!!!!!!!!!!!!!!!!!!!(This one is golden) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # v10_91 kept block=8 but add one transfomer block for class model + add 2 linear/dropout/relu 
    # v10_92/3/4 kept block=8 but add 3/6/9 linear/dropout/relu (no good) 
    # v10_95  kept block=8 but add one transfomer block for class model + add 3 linear/dropout/relu (no good) 
    # v10_96  same as v10_90 !!!!!!!!!!!!!!!!!!!!(This one is golden) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# v10_80 same as v10_60: units = 128; head = 4; layern = 8 (kept block=7) but using all training data + 0.002 40e 
# v10_82 same as v10_90: units = 128; head = 4; layern = 8 (kept block=8) but using all training data + 0.002 30e 
# v10_83 same as v10_90: units = 128; head = 4; layern = 8 (kept block=8) but using all training data + 0.001 30e 
# v10_84 same as v10_90: units = 128; head = 4; layern = 8 (kept block=8) but using all training data + 0.001 20e 
# v10_85 same as v10_90: units = 128; head = 4; layern = 8 (kept block=8) but using all training data + 0.001/2 30e no good
# v10_86 same as v10_90: units = 128; head = 4; layern = 8 (kept block=8) but using all training data + 0.001/5 30e no good worse than 10_85 

# v11_00 same as v10_96: units = 256; head = 8; layern = 4 (kept block=8) 40e 0.002
# v11_01 same as v11_00: but 30 e 0.002
# v11_02 same as v11_00: but 30 e 0.001 (!the best!) 
# v11_03 same as v11_00: but 30 e 0.001 units = 256; head = 8; layern = 8 
# v11_04 same as v11_00: but 30 e 0.001/2 units = 256; head = 8; layern = 8
# v11_05 same as v11_00: but 50 e 0.001/2 units = 256; head = 8; layern = 8
# v11_06 same as v11_05: but 50 e 0.001/2 units = 256; head = 8; layern = 8 all the data 
# v11_07 same as v11_06 but test randomness
# v11_08 same as v11_06 but 20e 0.001
# v11_09 same as v11_06 but 30e 0.001
# v11_10 same as v11_06 but 10e 0.001
# v11_20 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 128, 4, 8 ! The best
# v11_21 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 256, 8, 8
# v11_22 30e 0.001, but filter traning data using x,y step 4: with 1/16 samples annd use 256, 8, 8
# v11_23 30e 0.001, but filter traning data using x,y step 2: with 1/4  samples annd use 128, 4, 8
# v11_24 30e 0.001, but filter traning data using x,y step 3: with 1/9  samples annd use 128, 4, 8
# v11_25 30e 0.001, but filter traning data using x,y step 4: with 1/16 samples annd use 128, 4, 8
# v11_26 30e 0.001, but filter traning data using x,y step 6: with 1/36 samples annd use 128, 4, 8
# v11_27 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 128, 4, 8, same as 11.20 (not the best) 
# v11_28 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 256, 8, 6
# v11_29 30e 0.001, but filter traning data using x,y step 4: with 1/16 samples annd use 256, 8, 6
# v11_30 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 256, 8, 4
# v11_31 30e 0.001, but filter traning data using x,y step 4: with 1/16 samples annd use 256, 8, 4
# v11_32 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 512, 8, 6
# v11_33 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 128, 4, 12
# v11_34 30e 0.001, but filter traning data using x,y step 5: with 1/25 samples annd use 128, 4, 6

# v11_35 same as v11_20 but using pre-train mean & std 
# v11_36 same as v11_24 but using pre-train mean & std 
# v11_37 same as v11_25 but using pre-train mean & std 
# v11_38 same as v11_27 but using pre-train mean & std (mean & std does not matter)
# v11_39 same as v11_34 but using pre-train mean & std (mean & std does not matter)
# v11_40 same as v11_20 but using pre-train mean & std + all use the same pre-training 
# v11_41 same as v11_27 but using pre-train mean & std + all use the same pre-training 

# v11_50 same as v11_41 but n>3 

#**************************************************************************************************************************************************************************************************
# v11_60 process data with DEM and LST and add a control whether pre-training is needed 
# v11_61 Apr 18 2023 with LST data 
# v11_62 with LST data and local time data (no good) 
# v11_63 with LST data and local time data/10 (no good) 
# v11_64 with LST data and day of year: 2 variables (no good) 
# v11_65 with LST data and day of year: 2 variables and DEM 
# v11_66 with LST data and day of year: 1 variable  and DEM (little help) 
# v11_67 with LST data and day of year: 1 variable  and DEM, slope (the best ! ) - better than pre-training 
# v11_68 with LST data and day of year: 1 variable  and DEM, slope, aspect (does not help ) 
# v11_67 with LST data and day of year: 1 variable  and DEM, slope -  drop thermal
#**************************************************************************************************************************************************************************************************
# v9_8/v9_9* on Feb 18 2023 to use reflectance pre-training 
# v9_90 using pre-train; v9_91 include testing data in pre-train; 
# v9_92 using training and only >7 data in pre-train; # v9_93 only >7 data in pre-train & 0.0001 rate (no good) 
# v9_94 only >7 data in pre-train & 50 epochs 
# v9_95 only >7 data in pre-train + 50 epochs in fine tune + cls token
# v9_96 cls token without pretraining 
# v9_97 pre-training has 1-more layer & fixed a bug in copying parameters - 1 time run not copied ! great version 
# v9_98  pre-training >7 and drop 3 obs. no improvement 
# v9_99  pre-training >7 and drop 1 obs. same as v9_94 
# v9_910 pre-training >7 and drop 1 obs. same as v9_97
# v9_911 pre-training >7 and drop 1 obs. same as v9_97, 70+30 e (not good) 
# ! v9_912 ! pre-training >7 and drop 1 obs. same as v9_97, 100 + 50 e
# v9_913/914 pre-training >7 and drop 1 obs. same as v9_97, 100 + 50 e + new pre-training dataset 

#**************************************************************************************************************************************************************************************************
# v9_7 on Feb 18 2023 to test no norm on band 1-6 (no norm is not as good as with norm) 
# ! v9_6 ! on Feb 18 2023 to test x,y as a time series variable better than before ! 

#************# v9_2 to v9_51 backup models 
# v9_4/v9_5/v9_51 on Feb 16, 2023 use metrics & CNN (4-layer 9_4, 5-layer 9_5), v9_51 only >7 observations 
# v9_3 on Feb 13, 2023 use daily without locations (and without sensors)
# v9_2 on Feb 13, 2023 use 16-day composite 
#************

# v9_0/v9_1 !! on Jan 20, 2023 using new dataset with four years v9.1 only select 3 years to process 
# v8_9 !! same as v8_8 but using all data for training 
# v8_8 !! golden !! same as v8_72 
# v8_77 same as v8_72 but no dropout (also )
# v8_74/v8_75/v8_76 test 6/5/4-layers but with 32 units in the in the xy non-temporal branch (not as good as before)  
# v8_7/v8_71/v8_72/v8_73 2/3/4/5-layers in the xy non-temporal branch (v8_72 and 4-layer is the best!)  
# v8_6/v8_7 !! golden !! use [0, 1, 2, 3] and embedding for the Landsat 4,5,7,8 sensors + add position encoder 
# v8_5 !! golden !! use [0, 1, 2, 3] and embedding for the Landsat 4,5,7,8 sensors
# v8_4 use [0, 0.1, 0.2, 0.4] for the Landsat 4,5,7,8 sensors
# v8_3 use [0, 0.1, 0.2, 1  ] for the Landsat 4,5,7,8 sensors
# v8_0/8_1/8_2 use [0,0,0,1 ] for the Landsat 4,5,7,8 sensors
# v8_2 tried different parameters
# v8_1 fixed sensor is not normalized in v8_0
# v8_0/v8_1 same as v7_6, i.e., daily, input is daily and 4D but converted to 16-day in model
# v7_8/7_9 same as v7_6, i.e., daily, on Dec 23, 2022 test for sensor encoder (v7_8 no sensor encoder but v7_9 with sensor encoder) input is daily and 4D but converted to 16-day in model
# v7_7 !!!!!! 3 years of data and 3 models together - note cnn learning rate is *10  !!!!! 
# v7_6 !!!!!! new attention for daily data                                           !!!!!

#**************************************************************************************************************************************************************************************************
# v1_0 to v7_5 including LSTM for history layers see v9_51
#**************************************************************************************************************************************************************************************************

# cd /gpfs/home/hankui.zhang/mycode.research/LCMAP/LCMAP_run
# module load python/3.7
# module load rasterionew
# module load libtiff 
# module load libgeotiff

import os
import sys
import logging
import socket
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

import train_test 
import customized_train
import model_partial
import transformer_encoder44

import importlib
print(socket.gethostname())
base_name = "this_log_"+socket.gethostname()

IS_TEST = 0 ## generate the model 
IS_TEST = 1 ## training and testing evaluation 

#*****************************************************************************************************************
## load csv file
## line interpotation file
csv_dir = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.DAILY.85.00.06.18.24997.sensor.csv'
new_file = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.csv"

csv_dir = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.DAILY.85.06.18.24997.sensor.st.dem.csv'
new_file = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.st.dem.csv"

## Apr 18, 2023 
csv_dir = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.DAILY.85.06.18.24997.sensor.st.dem.csv'
new_file = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.st.dem.v2.csv"
new_file_fixed_LST = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.st.dem.v2.lstfixed.csv"

pre_csv='/gpfs/scratch/dong.luo/lcmap/ards_rst/LCMAP_Landsat_ARD.DAILY.ALL.418.speed.yrs85.06.18.csv'
pre_file = "/gpfs/scratch/hankui.zhang/ARD_C2/dumped_csv_lcmap/LCMAP_CU_Landsat_ARD.DAILY.ALL3years.pretraining.csv"
pre_file = "/gpfs/scratch/hankui.zhang/ARD_C2/dumped_csv_lcmap/LCMAP_CU_Landsat_ARD.DAILY.ALL3years.subset.every5.csv"


# STEP = 1500
# FACTOR = 6
# arrayx = np.array (range(int(min(data_pre['x'])),int(max(data_pre['x']))+1,STEP*FACTOR)).astype(np.float32) 
# arrayy = np.array (range(int(min(data_pre['y'])),int(max(data_pre['y']))+1,STEP*FACTOR)).astype(np.float32) 
# indexx = np.in1d(data_pre['x'],arrayx)
# indexy = np.in1d(data_pre['y'],arrayy)
# indexxy = np.logical_and (indexx, indexy)
# indexxy.sum() 
# pre_file = "/gpfs/scratch/hankui.zhang/ARD_C2/dumped_csv_lcmap/LCMAP_CU_Landsat_ARD.DAILY.ALL3years.subset.every"+str(FACTOR)+".csv"
# data_pre[indexxy].to_csv(pre_file) 



class_field = 'label'
n_field2 = 'total_n'

if not os.path.exists(new_file):
    data_per_all = pd.read_csv(csv_dir)
    yclasses = data_per_all[class_field]
    ## this step by removing snow/ice class and total_n=0 plotids
    valid_index = np.logical_and.reduce((yclasses != 7, data_per_all[n_field2]>0, data_per_all['image_year']!=2000 )) 
    data_per_all[valid_index].to_csv(new_file) 

if not os.path.exists(pre_file):
    data_per_all = pd.read_csv(pre_csv)
    ## this step by removing snow/ice class and total_n=0 plotids
    valid_index = np.logical_and(data_per_all[n_field2]>0,data_per_all[n_field2]>0)
    data_per_all[valid_index].to_csv(pre_file) 


data_per = pd.read_csv(new_file)
data_pre = pd.read_csv(pre_file)
yclasses = data_per[class_field]

years = data_per['image_year']
valid_index = yclasses != 7

##*****************************************************************************************************************
## fix surface temperature 

if not os.path.exists(new_file_fixed_LST):
    index_blue = []
    for i in range(80):
        index_blue.append('{:03d}.blue'.format(i) )
    
    index_lst = []
    for i in range(80):
        index_lst.append('{:03d}.st'.format(i) )
    
    lst_all = np.array(data_per[index_lst ])
    lst_mean = lst_all[np.logical_and(lst_all!=0,np.logical_not (np.isnan(lst_all)))].mean() 
    data_per['cos_as'] = np.cos(data_per['aspect']*np.pi/180)
    data_per['sin_as'] = np.sin(data_per['aspect']*np.pi/180)
    # df.at[]
    # https://www.edureka.co/community/43222/python-pandas-dataframe-deprecated-removed-future-release
    for i in range (data_per.shape[0]):
        # print (data_per.iloc[i][index_blue])
        # print (data_per.iloc[i][index_lst ])
        lst = np.array(data_per.iloc[i][index_lst ]).astype(np.float32)
        indexi_blue = np.logical_not (np.isnan(np.array(data_per.iloc[i][index_blue]).astype(np.float32)) ) 
        indexi_lst  = np.logical_and (np.logical_not (np.isnan(lst)),lst!=0) 
        indexi2 = np.logical_and (indexi_blue, lst==0) 
        # break 
        if indexi2.sum()>0:
            if indexi_lst.sum()>0:
                # data_per.iloc[i][index_lst[indexi2]] = lst[indexi_lst].mean() 
                for jj in np.array(index_lst)[indexi2]:
                    data_per.at[i,jj] = lst[indexi_lst].mean() 
                
            else:
                print ("data_per.iloc[i] is empty for lst ")
                # data_per.iloc[i][np.array(index_lst)[indexi2]] = lst_mean
                # data_per[np.array(index_lst)[indexi2]][i]
                # data_per.set_value(i, np.array(index_lst)[indexi2], lst_mean)
                for jj in np.array(index_lst)[indexi2]:
                    data_per.at[i,jj] = lst_mean
                # data_per.at[i,np.array(index_lst)[indexi2].tolist()]
            # break 
    
    data_per.to_csv(new_file_fixed_LST) 
else:
    data_per = pd.read_csv(new_file_fixed_LST)

##*****************************************************************************************************************
## parameters 
minimal_N = 3 
PRE_TRAIN = False 
KEPT_BLOCK_FT = 8; pre_train_version = "v11_51"
units = 64; head = 4; layern = 3
units = 128; head = 4; layern = 8
# units = 256; head = 8; layern = 8
# units = 256; head = 8; layern = 4
# units = 512; head = 8; layern = 6
if PRE_TRAIN==False: 
    units = 64; head = 4; layern = 3

IMG_HEIGHT2 = 80   ; IMG_WIDTH2 = 7; IMG_BANDS2=1

BATCH_SIZE = 512;
YEARS = [1985, 2000, 2018]
YEARS = [1985, 2006, 2018]
# YEARS = [1985, 2000]
# LEARNING_RATE = 0.01; layer_n = 4; EPOCH = 5; ITERS = 1; L2 = 1e-3; METHOD=0; PERCENT = 0.1; GPUi = 0
LEARNING_RATE = 0.001; LAYER_N = 4; EPOCH = 5; ITERS = 1; L2 = 1e-3; METHOD=0; GPUi = 0


DROP = 0.1

MAX_L_IN16DAY = 4
PERIODS = 23 
DAYS = 16 
yr1 = 2000

#*****************************************************************************************************************
## split training & testing data with 80% for train and 20% for test
import train_test 
importlib.reload(train_test)
orders = train_test.random_split(data_per.shape[0],split_n=10)
orders_pre = train_test.random_split(data_pre.shape[0],split_n=10)
index_train = np.logical_and(orders>1 ,valid_index) # this index_train will be replaced later
index_test =  np.logical_and(orders<=0,valid_index)
unique_yclass = np.unique(yclasses)
print (np.unique(yclasses[index_train]))
print (np.unique(yclasses[index_test ]))
N_CLASS = np.unique(yclasses[index_train]).size

## check different years in training
print ("check different years in training split ")
for yeari in np.unique(years[index_train]):
    print ((years[index_train]==yeari).sum())


MODEL_DIR = "./model/"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

##*****************************************************************************************************************
## main function
if __name__ == "__main__":
    LEARNING_RATE = 0.001
    EPOCH = 200 
    EPOCH = 6
    METHOD = 2 # Hank
    ITERS = 1
    if IS_TEST==1:        ## 0 generate the model and 1 80/20 train/test
        ITERS=5 
    
    GPUi = 0
    print ("sys.argv n: " + str(len(sys.argv)))
    
    
    ##***************************************************
    ## input parameters 
    DROP            = float(sys.argv[1])
    EPOCH           =   int(sys.argv[2] )
    METHOD          =   int(sys.argv[3] )
    LEARNING_RATE   = float(sys.argv[4])
    # ITERS         = int(sys.argv[4] )
    L2              = float(sys.argv[5])
    if len(sys.argv)>6:
        BATCH_SIZE       = int(sys.argv[6])    
    
    if len(sys.argv)>7:
        GPUi       = int(sys.argv[7])

    if len(sys.argv)>8:
        YEARS       = int(sys.argv[8])
        
    print ("BATCH_SIZE" + str(BATCH_SIZE))
    #*****************************************************************************************************************
    ## set GPU
    if '__file__' in globals():
        # base_name = os.path.basename(__file__)+socket.gethostname()
        base_name = os.path.basename(__file__)[10:]
        print(os.path.basename(__file__))
    
    yr1=YEARS
    if isinstance(YEARS,list):
        yr1="all"
    
    # base_name = base_name+'.year'+str(yr1)+'.layer'+str(LAYER_N)+'.dim'+str(IMG_HEIGHT2)+'.METHOD'+str(METHOD)+'.LR'+str(LEARNING_RATE)+'.EPOCH'+str(EPOCH)+'.L2'+str(L2)
    base_name = base_name+'.year'+str(yr1)+'.layer'+str(LAYER_N)+'.dim'+str(IMG_HEIGHT2)+'.METHOD'+str(METHOD)+'.LR'+str(LEARNING_RATE)+'.L2'+str(L2)
    # base_name = base_name+'daily.model.year'+str(yr1)
    model_pre_name = MODEL_DIR+base_name+'.pre.model.h5'
    # print (base_name[9:15])
    model_pre_name = model_pre_name.replace(base_name[9:15],pre_train_version)
    model_pre_name = model_pre_name.replace('.year'+str(yr1),'.year'+"all" )
    print (model_pre_name)

    if METHOD==0:
        logging.basicConfig(filename=base_name+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    print (GPUi)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and 'subset' in pre_file:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[GPUi], 'GPU')  
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)    
    
    print(tf.config.get_visible_devices())
    logging.info (tf.config.get_visible_devices()[0])
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] 
    #*****************************************************************************************************************
    ## get train and testing data    ## with sensor as input
    importlib.reload(train_test)
    IMG_WIDTH2 = 8
    XY_DIM_N = 4
    proportion=1.0
    if IS_TEST==1:
        proportion=0.8
    
    IMG_HEIGHT2_pre = 80 
    y_train_d,y_test_d, input_pre_train_norm2,_,_,_,_, mean_train_pre, std_train_pre,_,_,training_location_pre,_  = \
        train_test.get_training_test_com2(data_pre,orders_pre,orders_pre>-1,IMG_HEIGHT2_pre,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=1.0, total_days=IMG_HEIGHT2_pre, use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)
    
    IMG_HEIGHT2=80
    IMG_WIDTH2 = 8
    importlib.reload(train_test)
    if PRE_TRAIN==False: 
        y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test,training_location,testing_location = \
            train_test.get_training_test_com_lst(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=proportion, total_days=80, 
            use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)
    else:
        y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test,training_location,testing_location = \
            train_test.get_training_test_com2_use_mean_std(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,mean_train_pre,std_train_pre,class_field,proportion=proportion, total_days=80, use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)
    
    # aa = input_images_train2[100,:,:,0]
    # ab = np.array(data_per[train_metric])
    # ab[index_train,:][100,:]

    ## convert to transform format
    print(f"training data (2d) shape: {input_images_train_norm2.shape}")  # (N, 6, 23, 1)
    train_n = input_images_train_norm2.shape[0]
    test_n  = input_images_test_norm2 .shape[0]
    
    def assign_sensor_code_xy(input_images_train_norm2, training_location,IMG_WIDTH2=IMG_WIDTH2,SENSOR_INDEX=7, location_n=2):
        masks = input_images_train_norm2[:,:,:,1].copy()
        data1 = input_images_train_norm2[:,:,:,0].copy()
        data1[masks==0] = -9999.0
        input_images_train_norm3 = np.full([data1.shape[0], data1.shape[2], IMG_WIDTH2+location_n], fill_value=-9999.0, dtype=np.float32)
        input_images_train_norm3[:,:,:IMG_WIDTH2] = np.moveaxis(data1,1,2)  
        for ii in range(location_n):
            input_images_train_norm3[:,:,IMG_WIDTH2+ii]  = training_location[:, (ii):(ii+1)] 
        
        sensors = [4,5,7,8]
        sensor_codes = [0, 1, 2, 3] # v8.5
        for si,sensori in enumerate(sensors):
            index_sensor = np.logical_and(input_images_train_norm3[:,:,SENSOR_INDEX]==sensori, input_images_train_norm3[:,:,SENSOR_INDEX] !=-9999)
            input_images_train_norm3 [index_sensor, SENSOR_INDEX] = sensor_codes[si]           
        
        return input_images_train_norm3
    
    input_images_train_norm3 = assign_sensor_code_xy(input_images_train_norm2, training_location, SENSOR_INDEX=7, location_n=XY_DIM_N)
    input_images_test_norm3  = assign_sensor_code_xy(input_images_test_norm2 ,  testing_location, SENSOR_INDEX=7, location_n=XY_DIM_N)
    input_pre_train_norm3    = assign_sensor_code_xy(input_pre_train_norm2 ,  training_location_pre,IMG_WIDTH2=8)
    
    #*****************************************************************************************************************
    # save mean and std 
    mean_name=MODEL_DIR+base_name+'.lcmap_year' +str(yr1) + '_mean.csv'
    mean = mean_train2.copy()
    std  = std_train2.copy()
    # !!!!!!! transpose & reshape are different 
    arr = np.concatenate((mean.reshape(1,mean.shape[0]*mean.shape[1]), std.reshape(1,mean.shape[0]*mean.shape[1]) )).transpose()
    header = 'mean,std'
    np.savetxt(mean_name, arr, fmt="%s", header=header, delimiter=",")   
    #*****************************************************************************************************************    
    # testi = 0
    YEARS_LIST = YEARS
    if isinstance(YEARS,list):
        trainx_transformer = input_images_train_norm3
        testx_transformer  = input_images_test_norm3 
        trainy_transformer = y_train
        trainx_cnn = input_images_train_norm2
        trainy_cnn = y_train
        trainx_transformer_pre_x = input_pre_train_norm3
    else:
        train_sub_index = (years==YEARS)[index_train]
        test_sub_index  = (years==YEARS)[index_test ]
        trainx_transformer = input_images_train_norm3[train_sub_index,:,:]
        testx_transformer  = input_images_test_norm3 [test_sub_index ,:,:]
        trainy_transformer = y_train[train_sub_index]
        trainx_cnn = input_images_train_norm2[train_sub_index,:,:,:]
        trainy_cnn = y_train[train_sub_index]
        YEARS_LIST = [YEARS]    
        train_sub_index_pre = (data_pre['image_year']==YEARS)
        trainx_transformer_pre_x = input_pre_train_norm3[train_sub_index_pre,:,:]
    
    train_n = trainx_transformer.shape[0]   
    per_epoch = train_n//BATCH_SIZE
    print ("Train n = " + str(train_n) )
    #*****************************************************************************************************************
    print ("\n\n#partical CNN and transformers*************************************************************************\n\n\n")
    accuracylist1 = list()
    importlib.reload(model_partial)
    import customized_train_lr
    importlib.reload(customized_train_lr)
    #*****************************************************************************************************************
    ## model 1  13 * 3
    accuracylist2 = list()    
    
    #*****************************************************************************************************************
    ## model any length transformer 
    drop = DROP
    layern_ref = layern
    import customized_train_lr
    importlib.reload(customized_train_lr)
    per_epoch = train_n//BATCH_SIZE
    validation_split = 0
    if IS_TEST==1:
        validation_split=0.04
    
    ### new datasets 
    # train_x_com = np.concatenate((trainx_transformer,testx_transformer))
    # big_than1_index = (train_x_com[:,:,0]!=-9999.0).sum(axis=1)>1
    # trainx_transformer_pre = train_x_com[big_than1_index,:,:].copy()
    big_than1_index = (trainx_transformer_pre_x[:,:,0]!=-9999.0).sum(axis=1)>=minimal_N 
    trainx_transformer_pre = trainx_transformer_pre_x[big_than1_index,:,:].copy()
    strategy = tf.distribute.MirroredStrategy()
    # exit()
    layer_n = 5
    training_times = []
    testing_times = []
    import time
    for i in range(ITERS):
        print_str = "\n {:3d}: transformer model ********************************************************************************iter".format(i+1)
        print (print_str); logging.info (print_str)
        importlib.reload(transformer_encoder44)
        ## pre-training started
        if os.path.isfile(model_pre_name) and PRE_TRAIN: 
            model_ref = tf.keras.models.load_model(model_pre_name, compile=False)    
            # model_ref = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=trainx_transformer_pre.shape[1],n_feature=IMG_WIDTH2-2,n_out=N_CLASS, layern=layern_ref, units=units, n_head=head, drop=drop,is_day_input=True,is_sensor=True, is_sensor_embed=True, is_xy=True, is_reflectance=True)         
        else:
            with strategy.scope():
                model_ref = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=trainx_transformer_pre.shape[1],n_feature=IMG_WIDTH2-2,n_out=N_CLASS,\
                    layern=layern, units=units, n_head=head, drop=drop,is_day_input=True,is_sensor=True, is_sensor_embed=True, is_xy=True, xy_n=XY_DIM_N, is_reflectance=True)         
            
            importlib.reload(customized_train)   
            N_pre = trainx_transformer_pre.shape[0]
            if  N_pre>0 and PRE_TRAIN: 
                per_epoch_pre = N_pre//BATCH_SIZE
                customized_train.trainings_val_ref_with_lamda_1schedule(model_ref,trainx_transformer_pre,mean_train_pre, std_train_pre,strategy, BATCH_SIZE=BATCH_SIZE,\
                    start_rate=LEARNING_RATE,epochs=40,per_epoch=per_epoch_pre,split_epoch=5,option=METHOD,decay=L2)
                # sub_train = trainx_transformer_pre[:N_pre:500,:,:]
                # customized_train.trainings_val_ref_with_lamda_1schedule(model_ref,sub_train,mean_train_pre, std_train_pre,strategy, BATCH_SIZE=BATCH_SIZE,start_rate=LEARNING_RATE,
                    # epochs=EPOCH,per_epoch=sub_train.shape[0]//BATCH_SIZE,split_epoch=5,option=METHOD,decay=L2)
                # customized_train.trainings_val_ref(model_ref,sub_train,mean_train_pre, std_train_pre,BATCH_SIZE=BATCH_SIZE,start_rate=LEARNING_RATE,\
                    # epochs=EPOCH,per_epoch=sub_train.shape[0]//BATCH_SIZE,split_epoch=5,option=METHOD,decay=L2)            
        
            model_ref2 = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=trainx_transformer_pre.shape[1],n_feature=IMG_WIDTH2-2,n_out=N_CLASS,\
                layern=layern_ref, units=units, n_head=head, drop=drop,is_day_input=True,is_sensor=True, is_sensor_embed=True, is_xy=True, xy_n=XY_DIM_N, is_reflectance=True)         
            
            layers = model_ref2.layers if len(model_ref2.layers)<len(model_ref.layers) else model_ref.layers
            for il,ilayer in enumerate(layers):
                ilayer1 = model_ref2.layers[il] 
                ilayer2 = model_ref .layers[il] 
                name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic!='_']) 
                name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic!='_']) 
                if name_cls==name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
                    print ("\t"+ilayer.name, end=" ")
                    model_ref2.layers[il].set_weights (model_ref.layers[il].get_weights())
            
            model_ref2.save(model_pre_name)
            # test saved model 
            # model = tf.keras.models.load_model(model_name, compile=False)        
        
        ## ****************************************************************
        ## classification model construction
        model_drop = drop 
        # model_drop = 0 
        importlib.reload(transformer_encoder44)
        # model = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=input_images_train_norm3.shape[1],n_feature=IMG_WIDTH2-2,n_out=N_CLASS,\
            # layern=layern, units=units, n_head=head, drop=model_drop,is_day_input=True,is_sensor=True, is_sensor_embed=True, is_xy=True, xy_n=XY_DIM_N, dense_layer_n=PRE_TRAIN) 

        model = model_partial.get_model_cnn_1d (IMG_HEIGHT=IMG_WIDTH2+4,IMG_WIDTH=IMG_HEIGHT2,layer_n=layer_n,num_classes=N_CLASS, is_batch=True, drop=drop, is_pool=True) 
        if i==0:
            print (model.summary())        
        
        ## ****************************************************************
        ## transfer learning
        if PRE_TRAIN:
            print ("copy weights from pre-training ")
            # layers = model.layers if len(model.layers)<len(model_ref.layers) else model_ref.layers       
            # layers = model_ref.layers
            is_pass_multi_head = 0 
            is_layer_normalization = 0
            il1 = -1
            for il,ilayer in enumerate(model_ref.layers):
                if il1>= len(model.layers):
                    break
                
                # ilayer1 = model    .layers[il] 
                ilayer2 = model_ref.layers[il] 
                if (model_drop==0 and 'dropout' not in ilayer2.name) or model_drop>0: # to handle one model has dropout while the other does not 
                    il1=il1+1 
                else:
                    continue 
                
                ilayer1 = model    .layers[il1] 
                name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic!='_']) 
                name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic!='_']) 
                # print (name_cls + " " + name_ref)
                if "multi_head_attention" in ilayer.name:
                    is_pass_multi_head = is_pass_multi_head+1  
                
                if "layer_normalization" in ilayer.name:
                    is_layer_normalization = is_layer_normalization+1  
                
                if name_cls==name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
                    print ("\t"+ilayer.name, end=" ")
                    model.layers[il1].set_weights (model_ref.layers[il].get_weights())
                    
                    if is_pass_multi_head<=KEPT_BLOCK_FT:
                        # print ("\t"+ilayer.name)
                        model.layers[il1].trainable = False 
                
                if is_layer_normalization>=layern*2: # important if layer no are different
                    break; 
            
            print ('\nis_pass_multi_head = {:2d} and KEPT_BLOCK_FT = {:2d} and is_layer_normalization = {:2d} '.format(is_pass_multi_head, KEPT_BLOCK_FT, is_layer_normalization) )
        
        def transformx(trainx_transformer):
            trainx_transformer2 = trainx_transformer.copy()
            for ri in range(trainx_transformer.shape[0]):
                trainx_transformer2[ri,:,:] = -9999. 
                # print("")
                for di in range(IMG_HEIGHT2): 
                    doyi = trainx_transformer[ri,di,6];
                    if doyi==-9999.:
                        break
                    
                    doyi_in = min(int(doyi*80.0/366),79)
                    # print(doyi_in, end = "\t")
                    # if trainx_transformer2[ri,doyi_in,6]!=-9999.:
                        # doyi_in = min(doyi_in+1,79)
                    
                    while trainx_transformer2[ri,doyi_in,6]!=-9999.:
                        doyi_in=doyi_in+1
                        if doyi_in>=80:
                            doyi_in = 0 
                    
                    # print(doyi_in)
                    if trainx_transformer2[ri,doyi_in,6]!=-9999.:
                        print("!!!trainx_transformer2[ri,doyi_in,6]!=-9999.")
                        continue 
                    
                    # print(doyi_in)
                    trainx_transformer2[ri,doyi_in,:] = trainx_transformer[ri,di,:]
            
            doy_array = np.array(range(1,366)) 
            trainx_transformer2[:,:,6][trainx_transformer2[:,:,6]!=-9999.] = (trainx_transformer2[:,:,6][trainx_transformer2[:,:,6]!=-9999.]-doy_array.mean() )/doy_array.std ()
            trainx_transformer2[trainx_transformer2==-9999.] = 0
            return trainx_transformer2
        ## fine-tuning 
        start = time.time()
        trainx_transformer2 = transformx(trainx_transformer)
        importlib.reload(customized_train)  
        # model_history = customized_train.my_train_1schedule(model,trainx_transformer,trainy_transformer,epochs=10,start_rate=LEARNING_RATE,\
            # loss=loss,per_epoch=per_epoch,split_epoch=5,option=METHOD,decay=L2,batch_size=BATCH_SIZE,validation_split=validation_split)
        model_history = customized_train.my_train_1schedule(model,trainx_transformer2,trainy_transformer,epochs=EPOCH,start_rate=LEARNING_RATE,\
            loss=loss,per_epoch=per_epoch,split_epoch=5,option=METHOD,decay=L2,batch_size=BATCH_SIZE,validation_split=validation_split)
                
        end1 = time.time()
        for yeari in YEARS_LIST:
            print (yeari)
            test_sub_index = (years==yeari)[index_test]
            testx_transformer = input_images_test_norm3[test_sub_index,:,:]
            testx_transformer2 = transformx(testx_transformer)
            testy_transformer = y_test[test_sub_index]
            accuracy,classesi = customized_train.test_accuacy(model,testx_transformer2,testy_transformer)
            testx_index2 = dat_out['image_year']==yeari
            dat_out['predicted_cnn'+str(i)][testx_index2] = classesi
            # classesi
            print (">>>>>>>>>>>>>>>tranfatt" + '  {:0.4f}'.format(accuracy) )
            accuracylist2.append (accuracy)
        
        end2 = time.time()
        training_times.append(end1-start)
        testing_times.append(end2-end1)
        model_name = MODEL_DIR+base_name+'.model.h5'
    
    csv_name = MODEL_DIR+base_name+'.predict.csv'
    model.save(model_name)
    dat_out.to_csv(csv_name)
    #*****************************************************************************************************************
    ## model  LSTM 
    accuracylist3 = list()
    
    #*****************************************************************************************************************
    ## print accuacy 
    print (accuracylist1)
    print (accuracylist2)
    print (accuracylist3)
    i=0
    training_times = np.array(training_times)
    testing_times  = np.array(testing_times )
    print ("Training time m={:6.2f} sd={:6.2f} Testing time m={:6.2f} sd={:6.2f}".format(training_times.mean(),training_times.std(),testing_times.mean(),testing_times.std()))
    for yeari in YEARS_LIST: 
        acc_index = np.array(range(i,len(accuracylist1),len(YEARS_LIST)))
        if accuracylist1!=[] and acc_index.size>0:
            print ('{:4d}'.format(yeari)+" year accuracylist1 rf mean" + '  {:4.2f}'.format(np.array(accuracylist1)[acc_index].mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist1)[acc_index].std()*100) )
        
        acc_index = np.array(range(i,len(accuracylist2),len(YEARS_LIST)))
        if accuracylist2!=[] and acc_index.size>0:
            print ('{:4d}'.format(yeari)+" year accuracylist2 2d mean" + '  {:4.2f}'.format(np.array(accuracylist2)[acc_index].mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist2)[acc_index].std()*100) )
        
        acc_index = np.array(range(i,len(accuracylist3),len(YEARS_LIST)))
        if accuracylist3!=[] and acc_index.size>0:
            print ('{:4d}'.format(yeari)+" year accuracylist3 1d mean" + '  {:4.2f}'.format(np.array(accuracylist3)[acc_index].mean()*100) + "\nstd" + '  {:4.2f}'.format(np.array(accuracylist3)[acc_index].std()*100) )
        
        i=i+1


# pre_file = "/gpfs/scratch/hankui.zhang/ARD_C2/dumped_csv_lcmap/LCMAP_CU_Landsat_ARD.DAILY.ALL3years.pretraining_subset.csv"
# data_pre.iloc[:data_pre.shape[0]:30].to_csv(pre_file) 
# pre_csv='/gpfs/scratch/dong.luo/lcmap/ard_samples/LCMAP_CU_Landsat_ARD.DAILY.015006.yr85.06.18.csv'
# pre_file = "/gpfs/scratch/hankui.zhang/ARD_C2/dumped_csv_lcmap/LCMAP_CU_Landsat_ARD.DAILY.h15v06.pre2.csv"


 
