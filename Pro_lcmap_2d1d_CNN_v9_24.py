"""
Adapted on Mon Oct 17 09:54:57 2022
@author: Dong.Luo
The code is based on the "Pro_2d1d_CNN_v7_4.py" from NLCD pixel classification 
class order: developed=0, cropland=1, grass/shrub=2, tree cover=3, water=4, wetland=5, barren=6, snow/ice=7
"""
# v2_9 Dec 01, 2022 partical_cnn and tranformer; 
"""
partical-cnn: change to 4 layer
transformer-att: average pool + mask(update), active = softmax, units = 128
transformer-cls: chanage active to softmax, units = 128
"""
# v9_24 same as v9_23 but without using pooling (slightly worse than v9_23)
# v9_23 (used in the paper) based on 16-day composite (interpolated) but using CNN on Dec 26, 2023 for paper revision
# v9_21 (used in the paper) duplicate v9_2, i.e., based on 16-day composite (interpolated), on Dec 26, 2023 for paper revision
# v9_2  (used in the paper) use 16-day composite 
# v9_0 !! on Jan 20, 2023 using new dataset with four years v9.1 only select 3 years to process 
# v8_9 !! same as v8_8 but using all data for training 
# v8_8 !! golden !! same as v8_72 
# v8_77 same as v8_72 but no dropout (also )
# v8_74/v8_75/v8_76 test 6/5/4-layers but with 32 units in the in the xy non-temporal branch 
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

# LSTM v7_5 LSTM best parameters - batch=512, AdamW=1e-4,rate=0.001/0.01,dropout and layernorm work. units=64
# CNN  v7_4 best parameters, 4-layer, drop=0.5, batch=512, units=[128, 128, 256], AdamW=1e-4,rate=0.01/0.001 (0.01 is slightly better)
# v6_5 use [3,7] as convolution
# v6_4 same as v6_3
# v6_2 one-layer drop and v6_3 2-layer drop very good
# v5_9, 6_0, 6_1 testing cnn only data Dec 03, 2022 same as v5_6 using 2 years of data

# v5_7 Dec 03, 2022 same as v5_6 using 2 years of data
# v5_6 Dec 03, 2022 back to drop=0.1, batch=512, layer=3, head=4, units=64, AdamW=1e-4,rate=0.001
# v5_5 Dec 03, 2022 test for drop is 0: drop is very important !!!!!!!!!!!!!
# v5_4 Dec 03, 2022 test for 4-layers 
# v5_3 Dec 03, 2022 test for batch size 
# v5_1 Dec 03, 2022 test for 4-head and 0.3 drop
# v5_0 Dec 03, 2022 test for 8-head 
# v4_9 Dec 03, 2022 transformer unit 64 and ff 128 instead of 256
# v4_8 Dec 03, 2022 transformer unit 64 and ff 128 instead of 256
# v4_7 Dec 03, 2022 test for tf.math.divider in transformer and softmax in cnn and transformer unit 32
# v4_6 Dec 02, 2022 test for 12-layer cnn and 256-layer
# v4_5 Dec 02, 2022 test for 8-layer cnn and 128-layer
# v4_4 Dec 02, 2022 test for no relu in embedding
# v1_3 Nov 29, 2022 partical_cnn and tranformer; "dat_out['predicted_cnn2']= partical_cnn"; "dat_out['predicted_cnn3']= tranf-att"; "dat_out['predicted_cnn4']= tranf-cls"
# v1_2 Nov 28, 2022 1d cnn 
# v1_1 Oct 21, 2022 make it clear based on lcmap
# v1_0 Oct 18, 2022 adapted the code based on "Pro_2d1d_CNN_v7.4" and make the model is running
#################################################################################
# Hank on Dec 23, 2021 on partial convolution test 
# v1.3-v7.3 refer to original code
# v7.4 on Mar 05, 2022 to use 5 metrics & 5 observation threshold 
#################################################################################
# date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
# python Pro_learn_partial_CNN_v1_5.py
# date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
# time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
# date
# echo "$time_diff hours used";

# import datetime
# start_time = datetime.datetime.now()
# import Pro_2d1d_CNN_v6_0
# end_time = datetime.datetime.now()
# print("Used time: "+'{:5.2f}'.format((end_time-start_time).seconds/3600+(end_time-start_time).days*24) +"  hours")
# print(end_time)

# import datetime
# start_time = datetime.datetime.now()
# import importlib
# importlib.reload(Pro_2d1d_CNN_v2_3)
# end_time = datetime.datetime.now()
# print("Used time: "+'{:5.2f}'.format((end_time-start_time).seconds/3600+(end_time-start_time).days*24) +"  hours")
# print(end_time)

# module load cuda
# module load cudnn
# module load python/3.7
# module load rasterio
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
# import customized_train
import model_partial
import transformer_encoder44

# import plot_time_series
# from plot_time_series import band_fields


import importlib
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
print(socket.gethostname())
base_name = "this_log_"+socket.gethostname()

IS_TEST = 0 ## generate the model 
IS_TEST = 1 ## training and testing evaluation 

#*****************************************************************************************************************
## load csv file
## line interpotation file
csv_dir = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.DAILY.85.00.06.18.24997.sensor.csv'
yr1 = 2000
# metric: /gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.16DAY.metric.85.00.06.18.24997.csv
# daily: /gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.DAILY.85.00.06.18.24997.sensor.csv


# csv_metric = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.16DAY.metric.itp.85.00.06.18.24997.csv'
# data_met_all = pd.read_csv(csv_metric)


csv_metric = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.16DAY.metric.itp.85.00.06.18.24997.csv'
csv_dir = '/gpfs/scratch/dong.luo/lcmap/features/LCMAP_CU_Landsat_ARD.16DAY.metric.itp.85.00.06.18.24997.csv'
# data_met_all = pd.read_csv(csv_metric)

new_file = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.csv"
new_file = "./LCMAP_CU_Landsat_ARD.16-day.metric.no.ice.sensor3years.csv"

class_field = 'label'
n_field2 = 'total_n'

if not os.path.exists(new_file):
    data_per_all = pd.read_csv(csv_dir)
    # data_per_all_yr = data_per_all[data_per_all['image_year']==yr1]
    yclasses = data_per_all[class_field]
    # valid_index = np.logical_and(yclasses != 12, data_per_all[n_field2]>11)
    ## this step by removing snow/ice class and total_n=0 plotids
    valid_index = np.logical_and.reduce((yclasses != 7, data_per_all[n_field2]>0, data_per_all['image_year']!=2000 )) 
    data_per_all[valid_index].to_csv(new_file) 

data_per = pd.read_csv(new_file)
yclasses = data_per[class_field]

years = data_per['image_year']
# valid_index = np.logical_and(yclasses != 12, data_per[n_field2]>11)
valid_index = yclasses != 7

#*****************************************************************************************************************
## split training & testing data with 80% for train and 20% for test
import train_test 
importlib.reload(train_test)
# orders = train_test.random_split(data_per.shape[0],rate=0.2)
orders = train_test.random_split(data_per.shape[0],split_n=10)
index_train = np.logical_and(orders>1 ,valid_index)
index_test =  np.logical_and(orders<=0,valid_index)
unique_yclass = np.unique(yclasses)
print (np.unique(yclasses[index_train]))
print (np.unique(yclasses[index_test ]))
N_CLASS = np.unique(yclasses[index_train]).size

## check different years in training
print ("check different years in training split ")
for yeari in np.unique(years[index_train]):
    print ((years[index_train]==yeari).sum())

##*****************************************************************************************************************
## construct training and testing data
## plain CNN model
## 16 days metric with line interpotation (IMG_HEIGHT is time, IMG_WIDTH is featrue)
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

MODEL_DIR = "./model/"
if not os.path.isdir(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if __name__ == "__main__":
    LEARNING_RATE = 0.001
    EPOCH = 200 
    EPOCH = 6
    METHOD = 2 # Hank
    ITERS = 1
    if IS_TEST==1:
        ITERS=5 
        ## generate the model 
    
    # PERCENT = 0.1
    GPUi = 0
    print ("sys.argv n: " + str(len(sys.argv)))
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
    base_name = base_name+'daily.model.year'+str(yr1)
    if METHOD==0:
        logging.basicConfig(filename=base_name+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    print (GPUi)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[GPUi], 'GPU')  
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)    
    
    print(tf.config.get_visible_devices())
    logging.info (tf.config.get_visible_devices()[0])
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] 
    #*****************************************************************************************************************
    ## get train and testing data
    importlib.reload(train_test)
    ## with sensor as input
    IMG_WIDTH2 = 6
    IMG_HEIGHT2 = 23
    # y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test = \
        # train_test.get_training_test_com2(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=0.8, total_days=80, use_day=True, is_single_norm=True, use_sensor=True)
    
    proportion=1.0
    if IS_TEST==1:
        proportion=0.8
    
    # y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test,training_location,testing_location = \
        # train_test.get_training_test_com2(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=proportion, total_days=23, use_day=False, is_single_norm=True, use_sensor=False, use_xy=False)

    # y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test = \
        # train_test.get_training_test_com2(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=proportion, total_days=23, use_day=False, is_single_norm=True, use_sensor=False, use_xy=False)

    
    y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test = \
        train_test.get_training_test_com2(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion = 0.8)
    # aa = input_images_train2[100,:,:,0]
    # ab = np.array(data_per[train_metric])
    # ab[index_train,:][100,:]

    ## convert to transform format
    print(f"training data (2d) shape: {input_images_train_norm2.shape}")  # (N, 6, 23, 1)
    train_n = input_images_train_norm2.shape[0]
    test_n  = input_images_test_norm2 .shape[0]
    # change to shape (batch, time, feature) 
    # training 
    def assign_sensor_code(input_images_train_norm2):
        masks = input_images_train_norm2[:,:,:,1].copy()
        data1 = input_images_train_norm2[:,:,:,0].copy()
        data1[masks==0] = -9999.0
        input_images_train_norm3 = np.moveaxis(data1,1,2)  
        sensors = [4,5,7,8]
        sensor_codes = [0, 0.1, 0.2, 1] # v8.3 
        # sensor_codes = [0, 0.1, 0.2, 0.4] # v8.4 
        sensor_codes = [0, 1, 2, 3] # v8.5
        for si,sensori in enumerate(sensors):
            # print(si)
            # print(sensori)
            index_sensor = np.logical_and(input_images_train_norm3[:,:,7]==sensori, input_images_train_norm3[:,:,7] !=-9999)
            input_images_train_norm3 [index_sensor, 7] = sensor_codes[si]           
        
        return input_images_train_norm3
    
    # input_images_train_norm3 = assign_sensor_code(input_images_train_norm2)
    # input_images_test_norm3  = assign_sensor_code(input_images_test_norm2 )
    
    print(f"training data (2d) shape: {input_images_train_norm2.shape}")  # (N, 6, 23, 1)
    train_n = input_images_train_norm2.shape[0]
    test_n  = input_images_test_norm2 .shape[0]
    # change to shape (batch, time, feature)
    masks = input_images_train_norm2[:,:,:,1].copy()
    data1 = input_images_train_norm2[:,:,:,0].copy()
    data1[masks==0] = -9999.0
    input_images_train_norm3 = np.moveaxis(data1,1,2) 
    masks = input_images_test_norm2[:,:,:,1].copy()
    data1 = input_images_test_norm2[:,:,:,0].copy()
    data1[masks==0] = -9999.0
    input_images_test_norm3 = np.moveaxis(data1,1,2) 
    
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
        trainy_transformer = y_train
        trainx_cnn = input_images_train_norm2
        trainy_cnn = y_train
        # training_xy = training_location
    else:
        train_sub_index = (years==YEARS)[index_train]
        trainx_transformer = input_images_train_norm3[train_sub_index,:,:]
        trainy_transformer = y_train[train_sub_index]
        # training_xy = training_location[train_sub_index]
        trainx_cnn = input_images_train_norm2[train_sub_index,:,:,:]
        trainy_cnn = y_train[train_sub_index]
        YEARS_LIST = [YEARS]    
        train_n = trainx_transformer.shape[0]
    
    per_epoch = train_n//BATCH_SIZE
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
    units=64
    head = 4
    drop = 0   # 84.85%
    drop = 0.1 # 85.04%
    # drop = 0.2 # 84.63%
    # drop = 0.5 # 82.72% 
    # units=128 # not good see below
    # drop = DROP # 0     /0.1   /0.2    /0.3/0.4   /0.5  
    ##drop = DROP 83.02%/84.73%/84.94%/    /84.55%/83.71%
    drop = DROP
    import customized_train_lr
    importlib.reload(customized_train_lr)
    per_epoch = train_n//BATCH_SIZE
    validation_split = 0
    if IS_TEST==1:
        validation_split=0.04
    
    layer_n = 5
    training_times = []
    testing_times = []
    import time
    
    for i in range(ITERS):
        print_str = "\n 1: transformer model 23*6 ********************************************************************************iter" + str(i+1)
        print (print_str); logging.info (print_str)
        importlib.reload(transformer_encoder44)
        model = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=input_images_train_norm3.shape[1],n_feature=IMG_WIDTH2,n_out=N_CLASS,\
            layern=3, units=units, n_head=head, drop=drop,is_day_input=False,is_sensor=False, is_sensor_embed=False) 
        
        model = model_partial.get_model_cnn_1d (IMG_HEIGHT=IMG_WIDTH2,IMG_WIDTH=IMG_HEIGHT2,layer_n=layer_n,num_classes=N_CLASS, is_batch=True, drop=drop) 
        if i==0:
            print (model.summary())
        
        start = time.time()
        # model_history = customized_train_lr.my_train_schedule(model,[trainx_transformer, training_location],trainy_transformer,epochs=EPOCH,start_rate=LEARNING_RATE,\
            # loss=loss,per_epoch=per_epoch,split_epoch=5,option=METHOD,decay=L2,batch_size=BATCH_SIZE)
        model_history = customized_train_lr.my_train_schedule(model,trainx_transformer,trainy_transformer,epochs=EPOCH,start_rate=LEARNING_RATE,\
            loss=loss,per_epoch=per_epoch,split_epoch=5,option=METHOD,decay=L2,batch_size=BATCH_SIZE,validation_split=validation_split)
                
        end1 = time.time()
        for yeari in YEARS_LIST:
            print (yeari)
            test_sub_index = (years==yeari)[index_test]
            testx_transformer = input_images_test_norm3[test_sub_index,:,:]
            testy_transformer = y_test[test_sub_index]
            # testx_xy          = testing_location[test_sub_index,:]
            accuracy,classesi = customized_train_lr.test_accuacy(model,testx_transformer,testy_transformer)
            print (">>>>>>>>>>>>>>>tranfatt" + '  {:0.4f}'.format(accuracy) )
            accuracylist2.append (accuracy)
            
        end2 = time.time()
        training_times.append(end1-start)
        testing_times.append(end2-end1)
        model_name = MODEL_DIR+base_name+'.model.h5'
    
    # model.save(model_name)
    
    #*****************************************************************************************************************
    ## model  LSTM 
    accuracylist3 = list()
    
    #*****************************************************************************************************************
    ## print accuacy 
    print (accuracylist1)
    print (accuracylist2)
    print (accuracylist3)
    training_times = np.array(training_times)
    testing_times  = np.array(testing_times )
    print ("Training time m={:6.2f} sd={:6.2f} Testing time m={:6.2f} sd={:6.2f}".format(training_times.mean(),training_times.std(),testing_times.mean(),testing_times.std()))
    i=0
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

    #*****************************************************************************************************************
#     ## random forest
#     # if testi>=0:
#         # continue 
#     if True and layer_n==5:
#     # if False and layer_n==4:
#         clf = RandomForestClassifier(n_estimators=500) # 
#         clf.fit(input_images_train2[:,:,:,:IMG_BANDS2].reshape(train_n,IMG_HEIGHT2*IMG_WIDTH2*IMG_BANDS2), y_train.reshape(train_n))
#         classesi = clf.predict(input_images_test2[:,:,:,:IMG_BANDS2].reshape(test_n,IMG_HEIGHT2*IMG_WIDTH2*IMG_BANDS2))
#         accuracy = (y_test.reshape(test_n)==classesi).sum()/classesi.size
#         print ("rf" + str(testi) + '  {:0.4f}'.format(accuracy) )
#         print ("rf" + str(testi) + '  {:4.2f}'.format(accuracy*100) )
#         accuracylist3.append (accuracy)
#         print (accuracy)
#         dat_out['predicted_rf2' ] = classesi
    
    # file_name = MODEL_DIR+base_name+".csv"
    # dat_out.to_csv(file_name)
    
#     print (accuracylist1)
#     print (accuracylist2)
#     print (accuracylist3)


    
