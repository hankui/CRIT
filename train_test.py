"""
This code is hosting to convert csv data into:
    (1) split files
    (2)	get train and test data that fits model
"""
# import train_test 
import os 
import math 
import numpy as np  
import pandas as pd 

SPLIT_DIR = "./split/"

if not os.path.isdir(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

## *************************************************************************
## training and validation data split 
## invoked by customized_train.py
def random_split_train_validation (X_train,y_train,pecentage = 0.04):
    """
    split train into training and validating
    Used in the "customized_train.py" 
    """
    total_n = y_train.shape[0]
    sample_n = math.ceil(total_n*pecentage)
    split_n  = math.ceil(total_n/sample_n)
    file_index = SPLIT_DIR+"split.total_n"+str(total_n)+".for.validation.txt"
    
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    validation_index = orders==0
    training_index   = orders!=0
    sum(validation_index)
    sum(training_index  )
    return X_train[training_index],y_train[training_index],X_train[validation_index],y_train[validation_index],training_index,validation_index

## *************************************************************************
## training and testing data split 
## invoked by main function 
def random_split (total_n, split_n):
    """
    creat split index by split total_n into train and test based on split_n
    test: orders==0
    train: order!=0
    """
    sample_n = math.ceil(total_n/split_n)
    file_index = SPLIT_DIR+"index.total_n"+str(total_n)+".for.random.txt"
    if os.path.isfile(file_index):
        print ('file already exist! ' + file_index)
        dat = np.loadtxt(open(file_index, "rb"), dtype='<U10', delimiter=",", skiprows=1)
        orders = dat.astype(np.int64)
        
    else:
        header = 'order'
        orders = 0
        for i in range(split_n):
            if i==0:
                orders = np.repeat(i, sample_n)
            else:
                orders = np.concatenate((orders, np.repeat(i, sample_n)))
        
        orders = orders[range(total_n)]  
        np.random.shuffle(orders)
        np.savetxt(file_index, orders, fmt="%s", header=header, delimiter=",")
    
    return orders

####################################################################################
##########CORE FUCNTION TO GENERATE model input data################################
####################################################################################
# invoked by get_training_test_com2 function in this file 
# different from metrics - this one add masked layers to results for partial convolution to work 
# # IMG_HEIGHT = COMPOSITE_N; IMG_WIDTH=6; IMG_BANDS=1
# data_all = data_per
# train_fields = train_metric
# test_field = class_field
# IMG_HEIGHT = IMG_HEIGHT2
# IMG_WIDTH = IMG_WIDTH2
# IMG_BANDS = IMG_BANDS2; is_single_norm=True; 
# construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, is_single_norm=is_single_norm)
def construct_composite_train_test(data_all,index_train,index_test,train_fields,test_field,
                                   IMG_HEIGHT,IMG_WIDTH,IMG_BANDS,mean_train=0,std_train=1, 
                                   is_single_norm=False,is_train_test_com=True,use_lst=False,use_ltime=False):
    """
    Perpare train and test from composite csv file (simple version of 'construct_metric_train_test')
    input_images_train shape: (trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    creat a mask_train and mask_test
    mean_train,std_train shape: (IMG_WIDTH, 1,1) (1) when processing step50 data, 0 and 1, (2) when processing 25000, it used mean and std from step50 data
    """   
    trainx2 = np.array(data_all[train_fields][index_train]).astype(np.float32)
    input_images_train = trainx2.reshape(trainx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_train = np.array(data_all[test_field][index_train]).astype(np.int32)
    
    testx2 = np.array(data_all[train_fields][index_test]).astype(np.float32)
    input_images_test = testx2.reshape(testx2.shape[0],IMG_WIDTH,IMG_HEIGHT,IMG_BANDS)
    y_test = np.array(data_all[test_field][index_test]).astype(np.int32)
    train_n = input_images_train.shape[0]
    test_n  = input_images_test .shape[0]
    
    print(train_n)
    print(test_n )
    print(np.isnan(input_images_train).sum()/input_images_train.size*100)
    print(np.isnan(input_images_test ).sum()/input_images_test .size*100)
    
    ## check data
    # for i in range(input_images_train.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_train[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)
    
    # for i in range(input_images_test.shape[0]):
        # for j in range(IMG_HEIGHT):
            # spectra_ij = input_images_test[i,:,j,0]
            # if np.isnan(spectra_ij).all() or np.logical_not(np.isnan(spectra_ij)).all():
                # continue 
            # else:
                # print(i)
                # print(j)
                # print(spectra_ij)    
    ## change nan value to 0 
    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_BANDS)
    masks_train = np.logical_not(np.isnan(input_images_train)).astype(np.float32)
    input_images_train0 = np.concatenate((input_images_train,masks_train),axis=3)
    input_images_train0[np.isnan(input_images_train[:,:,:,0]),:] = 0
    masks_test = np.logical_not(np.isnan(input_images_test)).astype(np.float32)
    input_images_test0 = np.concatenate((input_images_test,masks_test),axis=3)
    input_images_test0[np.isnan(input_images_test[:,:,:,0]),:] = 0
    
    ## normalize 
    input_images_train_norm0 = input_images_train0.copy()
    input_images_test_norm0  = input_images_test0 .copy()
    
    ## this norm turn out to be very important 
    if is_train_test_com:
        a = np.ma.array(np.concatenate((input_images_train0[:,:,:,0],input_images_test0[:,:,:,0])), \
            mask=np.concatenate((input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0,input_images_test0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)))
    else:
        a = np.ma.array(input_images_train0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
        
    REF_BANDS_N = 6 + use_lst + use_ltime
    if is_single_norm==True:
        if isinstance(mean_train,int) and isinstance(std_train, int): 
            mean_train = a.mean(axis=(0,2)).reshape(a.shape[1],1,1)
            std_train  = a.std (axis=(0,2)).reshape(a.shape[1],1,1)
        # mean_train = 0
        # std_train  = 1
        input_images_train_norm0[:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_train0[:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
        input_images_test_norm0 [:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_test0 [:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
        ## input_images_train[0,:,:2,0]
        ## input_images_train_norm0[0,:,:2,0]
    elif is_single_norm==False:
        if isinstance(mean_train,int) and isinstance(std_train, int): 
            mean_train = a.mean(axis=0).reshape(a.shape[1],a.shape[2],1)
            std_train  = a.std (axis=0).reshape(a.shape[1],a.shape[2],1)
        # mean_train = 0
        # std_train  = 1
        input_images_train_norm0[:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_train0[:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
        input_images_test_norm0 [:,:REF_BANDS_N,:,:IMG_BANDS] = (input_images_test0 [:,:REF_BANDS_N,:,:IMG_BANDS] - mean_train[:REF_BANDS_N,:,:])/std_train[:REF_BANDS_N,:,:]
    else: # no normalize
        if isinstance(mean_train,int) and isinstance(std_train, int): 
            mean_train = a.mean(axis=(0,2)).reshape(a.shape[1],1,1)
            std_train  = a.std (axis=(0,2)).reshape(a.shape[1],1,1)
        input_images_train_norm0[:,:REF_BANDS_N,:,:IMG_BANDS] = input_images_train0[:,:REF_BANDS_N,:,:IMG_BANDS] 
        input_images_test_norm0 [:,:REF_BANDS_N,:,:IMG_BANDS] = input_images_test0 [:,:REF_BANDS_N,:,:IMG_BANDS] 
    
    # b = np.ma.array(input_images_train_norm0[:,:,:,0], mask=input_images_train0[:,:,:, IMG_BANDS:(IMG_BANDS+1)]==0)
    # b.mean(axis=0)
    # b.std(axis=0)
    return input_images_train_norm0,input_images_test_norm0,input_images_train0,input_images_test0,y_train,y_test,mean_train,std_train

###################################################
#######function: considering x, y, dem, slope######
###################################################
## *************************************************************************
# invoked by main function to normalize and split data for both 16-day composite and daily
## with training and testing index returned 
# total_days=23; use_day=False; proportion=0.8; 
# total_days=80; use_day=True; proportion=0.8; is_single_norm=False; use_sensor=True; use_xy=True
# fix a bug on Dec 28, 2022 to make sure that the training and testing are the same locations 
def get_training_test_com_lst(data_per,orders,valid_index,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2,class_field,proportion=0.1, total_days=23, use_day=False, \
        is_single_norm=False, use_sensor=False, use_xy=False, use_lst=False, use_ltime=False):
    """
    used construct_composite_train_test to get train and test
    "index.total_n74821.for.random.txt". Set proportion=0.8 and using 2018 plotids to get index_test and index_train
    """
    
    ## fix training and testing split bugs 
    ## if there are many years data - make sure the training samples are from the same locations 
    if 'image_year' in data_per.keys() and 'plotid' in data_per.keys():
        print ("! fixed a bug on 2022 12 28 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        unique_years = np.sort(np.unique(data_per['image_year']))
        ref_year = unique_years[-1]
        all_plot_ids = np.array(data_per['plotid'].copy() )
        if proportion==0.8:
            sub_index_train = np.logical_and.reduce((orders<8 ,valid_index, data_per['image_year']==ref_year))
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            index_train = np.logical_not(index_test)  
        elif proportion==1.0:
            print ("proportion==1.0:")
            # sub_index_train = np.logical_and.reduce((orders<8 ,valid_index, data_per['image_year']==ref_year))
            sub_index_test  = np.logical_and.reduce((orders>=8,valid_index, data_per['image_year']==ref_year))
            test_plotid = data_per[sub_index_test]['plotid']
            index_test  = np.in1d(all_plot_ids, test_plotid)
            index_train = np.logical_and(orders>-1,valid_index)
        else:
            print("shit I cannot come here")
    else:
        print ("!!!!! NOT 'image_year' in data_per.keys() and 'plotid' in data_per.keys()")
        if proportion==0.5:
            index_train = np.logical_and(orders<5 ,valid_index)
            index_test  = np.logical_and(orders>=5,valid_index)
        elif proportion==0.9:
            index_train = np.logical_and(orders<9 ,valid_index)
            index_test  = np.logical_and(orders>=9,valid_index)
        elif proportion==0.8:
            index_train = np.logical_and(orders<8,valid_index)
            index_test  = np.logical_and(orders>=8,valid_index)
        else:
            index_train = np.logical_and(orders==0,valid_index)
            index_test  = np.logical_and(orders!=0,valid_index)
    
    train_metric = list()
    bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2']
    if use_day:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy']
    
    if use_sensor:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'doy', 'sensor']

    if use_sensor and use_lst:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'doy', 'sensor']
        
    if use_sensor and use_lst and use_ltime:
        bandslist = ['blue', 'green', 'red', 'nir','swir1', 'swir2', 'st', 'ltime', 'doy', 'sensor']        
    
    for bandi in bandslist:
        # for ni in ('00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22'):    
        for nii in range(total_days):
            if use_day: ## this is correct as daily must use day as input 
                ni = '{:03d}'.format(nii)
            else: 
                ni = '{:02d}'.format(nii)
            
            train_metric.append(ni+'.'+bandi)
    
    input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,y_train,y_test,mean_train,std_train \
        = construct_composite_train_test(data_per,index_train,index_test,train_metric,class_field,IMG_HEIGHT2,IMG_WIDTH2,IMG_BANDS2, 
                                         is_single_norm=is_single_norm, is_train_test_com=proportion!=1.0, use_lst=use_lst, use_ltime=use_ltime)
        
    if use_xy:
        location_index = ['x','y','dem','slope','cos_as','sin_as']
        mean_x = np.array(data_per[location_index].mean())
        std_x  = np.array(data_per[location_index].std ())
        mean_train = np.concatenate((mean_train, mean_x.reshape(mean_x.size,1,1) ) )
        std_train  = np.concatenate((std_train , std_x .reshape(mean_x.size,1,1) ) )
        training_location = (np.array(data_per[index_train][location_index].copy())-mean_x)/std_x
        testing_location  = (np.array(data_per[index_test ][location_index].copy())-mean_x)/std_x        
    
    dat_out = pd.DataFrame()
    for propertyi in data_per.keys():
        if '0' not in propertyi:
            dat_out[propertyi] = (data_per[propertyi][index_test]).copy()
    
    dat_out['predicted_cnn0'] = 255
    dat_out['predicted_cnn1'] = 255
    dat_out['predicted_cnn2'] = 255
    dat_out['predicted_cnn3'] = 255
    dat_out['predicted_cnn4'] = 255
    
    if use_xy:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test,training_location,testing_location
    else:
        return y_train,y_test,\
            input_images_train_norm2,input_images_test_norm2,input_images_train2,input_images_test2,dat_out,mean_train,std_train,index_train,index_test

#########################################################################################################################################################################################
