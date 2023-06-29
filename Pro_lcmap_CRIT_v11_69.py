"""
This code is the application code for paper 
"Classifying raw irregular Landsat time series (CRIT) for large area land cover mapping by adapting Transformer model"
Three major processes:
    (1) load data and per-processing
    (2) manipulate data including creating training and testing data, make the data shape fits the model
    (3) model training and predcit.
"""
import os
import sys
import logging
import socket
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier

import train_test 
import customized_train
import transformer_encoder44

import importlib
print(socket.gethostname())
base_name = "this_log_"+socket.gethostname()

IS_TEST = 0 ## generate the model 
IS_TEST = 1 ## training and testing evaluation 

#*****************************************************************************************************************
## load csv file
csv_dir = './LCMAP_CU_Landsat_ARD.DAILY.85.06.18.24997.sensor.st.dem.csv'
new_file = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.st.dem.v2.csv"
new_file_fixed_LST = "./LCMAP_CU_Landsat_ARD.DAILY.metric.no.ice.sensor3years.st.dem.v2.lstfixed.csv"

####################################################################################################################################
class_field = 'label'
n_field2 = 'total_n'

if not os.path.exists(new_file):
    data_per_all = pd.read_csv(csv_dir)
    yclasses = data_per_all[class_field]
    ## this step by removing snow/ice class and total_n=0 plotids
    valid_index = np.logical_and.reduce((yclasses != 7, data_per_all[n_field2]>0)) 
    data_per_all[valid_index].to_csv(new_file) 

data_per = pd.read_csv(new_file)

yclasses = data_per[class_field]

years = data_per['image_year']
valid_index = yclasses != 7

##*****************************************************************************************************************
## fix surface temperature (working on 25000 data)
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
        lst = np.array(data_per.iloc[i][index_lst ]).astype(np.float32)
        indexi_blue = np.logical_not (np.isnan(np.array(data_per.iloc[i][index_blue]).astype(np.float32)) ) 
        indexi_lst  = np.logical_and (np.logical_not (np.isnan(lst)),lst!=0) 
        indexi2 = np.logical_and (indexi_blue, lst==0) 
        # break 
        if indexi2.sum()>0:
            if indexi_lst.sum()>0:
                for jj in np.array(index_lst)[indexi2]:
                    data_per.at[i,jj] = lst[indexi_lst].mean()                 
            else:
                print ("data_per.iloc[i] is empty for lst ")
                for jj in np.array(index_lst)[indexi2]:
                    data_per.at[i,jj] = lst_mean 
    
    data_per.to_csv(new_file_fixed_LST) 
else:
    data_per = pd.read_csv(new_file_fixed_LST)

##*****************************************************************************************************************
## parameters 
minimal_N = 3 
KEPT_BLOCK_FT = 8; pre_train_version = "v11_51"
units = 64; head = 4; layern = 3
IMG_HEIGHT2 = 80   ; IMG_WIDTH2 = 7; IMG_BANDS2=1
BATCH_SIZE = 512;
YEARS = [1985, 2006, 2018]
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
## out: 19918, 19997, 19940 
for yeari in np.unique(years[index_train]):
    print ((years[index_test]==yeari).sum())
## out: 2493, 2459, 2531
    
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
    
    base_name = base_name+'.year'+str(yr1)+'.layer'+str(LAYER_N)+'.dim'+str(IMG_HEIGHT2)+'.METHOD'+str(METHOD)+'.LR'+str(LEARNING_RATE)+'.L2'+str(L2)
    print (base_name[9:15])

    if METHOD==0:
        logging.basicConfig(filename=base_name+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    print (GPUi)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')        
    print(tf.config.get_visible_devices())
    logging.info (tf.config.get_visible_devices()[0])
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()] 
    #*****************************************************************************************************************
    ## get train and testing data   
    importlib.reload(train_test)
    IMG_WIDTH2 = 8
    XY_DIM_N = 4
    proportion=1.0
    if IS_TEST==1:
        proportion=0.8

    IMG_HEIGHT2=80
    IMG_WIDTH2 = 8
    importlib.reload(train_test)
    y_train,y_test, input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out, mean_train2, std_train2,index_train,index_test,training_location,testing_location = \
        train_test.get_training_test_com_lst(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,
                                             class_field,proportion=proportion, total_days=80, 
                                             use_day=True, is_single_norm=True, use_sensor=True, use_xy=True)

    ## convert to transform format
    print(f"training data (2d) shape: {input_images_train_norm2.shape}")  # (N, 8, 80, 2)
    train_n = input_images_train_norm2.shape[0]
    test_n  = input_images_test_norm2 .shape[0]
    
    def assign_sensor_code_xy(input_images_train_norm2, training_location,IMG_WIDTH2=IMG_WIDTH2,SENSOR_INDEX=7, location_n=2):
        """
		change sensor information into numerical numbers just use 3 sensors (5,7,8)
		change data shape from (N, 8, 80, 2) to (N, 80, 8+4). 1) remove mask; 2)add sensor information based on x and y
		"""
        masks = input_images_train_norm2[:,:,:,1].copy()
        data1 = input_images_train_norm2[:,:,:,0].copy()
        data1[masks==0] = -9999.0
        input_images_train_norm3 = np.full([data1.shape[0], data1.shape[2], IMG_WIDTH2+location_n], fill_value=-9999.0, dtype=np.float32)
        input_images_train_norm3[:,:,:IMG_WIDTH2] = np.moveaxis(data1,1,2)  
        for ii in range(location_n):
            input_images_train_norm3[:,:,IMG_WIDTH2+ii]  = training_location[:, (ii):(ii+1)] 
        
        sensors = [5,7,8]
        sensor_codes = [0, 1, 2] # v8.5
        for si,sensori in enumerate(sensors):
            index_sensor = np.logical_and(input_images_train_norm3[:,:,SENSOR_INDEX]==sensori, input_images_train_norm3[:,:,SENSOR_INDEX] !=-9999)
            input_images_train_norm3 [index_sensor, SENSOR_INDEX] = sensor_codes[si]                   
        return input_images_train_norm3  
    
    input_images_train_norm3 = assign_sensor_code_xy(input_images_train_norm2, training_location, SENSOR_INDEX=7, location_n=XY_DIM_N)
    input_images_test_norm3  = assign_sensor_code_xy(input_images_test_norm2 ,  testing_location, SENSOR_INDEX=7, location_n=XY_DIM_N)    
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
    #yr1 = [all, 1985, 2006, 2018]
    YEARS_LIST = YEARS
    if isinstance(YEARS,list):
        trainx_transformer = input_images_train_norm3
        testx_transformer  = input_images_test_norm3 
        trainy_transformer = y_train
        trainx_cnn = input_images_train_norm2
        trainy_cnn = y_train
    else:
        train_sub_index = (years==YEARS)[index_train]
        test_sub_index  = (years==YEARS)[index_test ]
        trainx_transformer = input_images_train_norm3[train_sub_index,:,:]
        testx_transformer  = input_images_test_norm3 [test_sub_index ,:,:]
        trainy_transformer = y_train[train_sub_index]
        trainx_cnn = input_images_train_norm2[train_sub_index,:,:,:]
        trainy_cnn = y_train[train_sub_index]
        YEARS_LIST = [YEARS]    
    
    train_n = trainx_transformer.shape[0]   
    per_epoch = train_n//BATCH_SIZE
    print ("Train n = " + str(train_n) )
    #*****************************************************************************************************************
    print ("\n\n#partical CNN and transformers*************************************************************************\n\n\n")
    accuracylist1 = list()
    #*****************************************************************************************************************
    ## model 1  13 * 3
    accuracylist2 = list()        
    #*****************************************************************************************************************
    ## model any length transformer 
    drop = DROP
    layern_ref = layern
    per_epoch = train_n//BATCH_SIZE
    validation_split = 0
    if IS_TEST==1:
        validation_split=0.04
        
    strategy = tf.distribute.MirroredStrategy()
    # exit()
    for i in range(ITERS):
        print_str = "\n {:3d}: transformer model ********************************************************************************iter".format(i+1)
        print (print_str); logging.info (print_str)
        importlib.reload(transformer_encoder44)

        ## ****************************************************************
        ## classification model construction (with 25000 data)
        model_drop = drop 
        # model_drop = 0 
        importlib.reload(transformer_encoder44)
        model = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=input_images_train_norm3.shape[1],n_feature=IMG_WIDTH2-2,n_out=N_CLASS,
                                                                                layern=layern, units=units, n_head=head, drop=model_drop,
                                                                                is_day_input=True,is_sensor=True, is_sensor_embed=True, is_xy=True, xy_n=XY_DIM_N)  # dense_layer_n=PRE_TRAIN
                                                                                
        if i==0:
            print (model.summary())        
        
        ## **************************************************************************************************************************************
        ## fine-tuning the model without transfer learning Or transfer learned model if PRE_TRAIN is True
        importlib.reload(customized_train)          
        model_history = customized_train.my_train_1schedule(model,trainx_transformer,trainy_transformer,epochs=EPOCH,start_rate=LEARNING_RATE,\
            loss=loss,per_epoch=per_epoch,split_epoch=5,option=METHOD,decay=L2,batch_size=BATCH_SIZE,validation_split=validation_split)
                
        for yeari in YEARS_LIST:
            print (yeari)
            test_sub_index = (years==yeari)[index_test]
            testx_transformer = input_images_test_norm3[test_sub_index,:,:]
            testy_transformer = y_test[test_sub_index]
            accuracy,classesi = customized_train.test_accuacy(model,testx_transformer,testy_transformer)
            testx_index2 = dat_out['image_year']==yeari
            dat_out['predicted_cnn'+str(i)][testx_index2] = classesi
            # classesi
            print (">>>>>>>>>>>>>>>tranfatt" + '  {:0.4f}'.format(accuracy) )
            accuracylist2.append (accuracy)
        
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
##################################################################################################################################################


 
