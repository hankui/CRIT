"""
This code is using trained model to predict land cover classes on each ARD tile
total 7 classes
PERIODS=80 the model input data is max=80
The code can handle number of observation higher 80
input data: 6 reflectance bands, doy, sensor, x, y, dem, slope
"""
# in ARD coordinates x<0, y>0
# a pixel in h18v18 (-152,264,  576,174)

import sys
import os 
import numpy as np 
import importlib 

import find_files
import Landsat_ARD_io
from datetime import datetime
import gc 
import rasterio 

import tensorflow as tf
import transformer_encoder44

np.set_printoptions(suppress=True) # turn off scientific notation  

BANDS_N = 6     # (b2-b7)
XY_DIM = 5000
FILLED = -32768

# MAX_L_IN16DAY = 4
# PERIODS = 23 
PERIODS = 80 
LONG_PERIODS = 140
# DAYS = 16 
BATCH = 4096 # 0:20:18.318266
BATCH = 4096*32 # ! not enough GPU !
BATCH = 4096*16 # ! not enough GPU !
BATCH = 4096*8  # ! not enough GPU !
BATCH = 4096*4  # 0:19:35.45517 not save much from 4096 

CLASS_DIR = "/class_results/"
# BROWSE_DIR = "/gpfs/scratch/hankui.zhang/ARD_C2/class_results/browse"

if not os.path.isdir(CLASS_DIR):
	os.makedirs(CLASS_DIR)

# if not os.path.isdir(BROWSE_DIR):
# 	os.makedirs(BROWSE_DIR)

## untared ARD direcotry and dem directory
## DEM file name example: re_projected_dem_h01v06.v1_4.tif 
INPUTDIR_ROOT = "/gpfs/scratch/untar/"
DEM_DIR = "/gpfs/scratch/dem/"

version='v2_0'
if '__file__' in globals():
    print(os.path.basename(__file__))
    version=os.path.basename(__file__)[24:28] 
    print(version)


model_path = "./model/2d1d_CNN_v11_69.py.yearall.layer4.dim80.METHOD2.LR0.001.L20.0001.model.h5"
mean_name='./model/2d1d_CNN_v11_69.py.yearall.layer4.dim80.METHOD2.LR0.001.L20.0001.lcmap_yearall_mean.csv'

x_mean = 0
x_std = 1
if os.path.exists(mean_name):
    dat_temp = np.loadtxt(open(mean_name, "rb"), dtype='<U30', delimiter=",", skiprows=1)
    arr = dat_temp.astype(np.float64)
    x_mean,x_std = arr[:,0],arr[:,1]
else:
    print("Error !!!! mean file not exists " + mean_name)


tile_id      = "005013"
tile_id      = "028004"
tile_id      = "007013"
tile_id      = "003012"
is_true_color = False
year = 2018 

if __name__ == "__main__":
    tile_id      = sys.argv[1]    
    if len(sys.argv)>2:
        year       = int(sys.argv[2])        
    if len(sys.argv)>3:
        is_true_color = bool(sys.argv[3])    
    
    print (year) 
    ## find all ARD files
    importlib.reload (find_files)
    QA_file_list = find_files.find_qa_list(INPUTDIR_ROOT+"/"+str(year), is_L8=False, pattern='_'+tile_id+'_')
    QA_file_list.sort()
    total_n = len(QA_file_list)
    print (str(total_n) + " files found for tile " + tile_id)
    
    if total_n==0:
        print("! There is no ARD files for this tile" + tile_id)
        exit()
    
    if total_n>PERIODS:
        print("Warning total_n>PERIODS=80 for this tile n=" + str(total_n))    
    
    ## find the dem file
    tile_id_format2 = 'h'+tile_id[1:3]+'v'+tile_id[4:]
    DEM_file_list = find_files.find_list(DEM_DIR+"/", is_L8=False, pattern=tile_id_format2)
    if len(DEM_file_list)!=1: 
        print ("! There is no DEM for this tile " + tile_id)
        print (DEM_file_list)
        exit()        
    
    dem_file = DEM_file_list[0] 
    
    ## find the class file
    class_file_list = find_files.find_list(CLASS_DIR+"/", is_L8=False, pattern=tile_id+"_"+str(year) )
    if len(class_file_list)>0: 
        print ("! The class file has been generated " + tile_id)
        print (class_file_list)
        # exit() # this need to be changed later 
    
    ## *************************************************************************************************************
    ## load data into memory 
    # sensor_codes = [0, 1, 2, 3] # v8.5
    sensor_labels = ["LT04", "LT05", "LE07", "LC08"]
    all_data = np.full([XY_DIM,XY_DIM,total_n,BANDS_N+2],fill_value=-9999.0,dtype=np.float32)
    all_qa   = np.full([XY_DIM,XY_DIM,total_n,         ],fill_value=False  ,dtype=bool   )
    doys    = np.full([total_n],fill_value=-9999.0,dtype=np.float32)
    sensors = np.full([total_n],fill_value=-9999.0,dtype=np.float32)
    sys.getsizeof(all_data)/1024/1024/1024 #GB
    
    importlib.reload(Landsat_ARD_io)
    for i in range(total_n):
        landsati = Landsat_ARD_io.Landsat_ARD_tile(QA_file_list[i],is_L8="LC08" in QA_file_list[i])        
        sensor_code = 0
        for li,label in enumerate(sensor_labels):
            if label in QA_file_list[i]:
                sensor_code = li 
                break
        
        print (i, end="\t") 
        landsati.load_data()
        all_data[landsati.is_valid,i,BANDS_N] = landsati.doy
        all_data[landsati.is_valid,i,BANDS_N+1] = sensor_code
        all_qa  [landsati.is_valid,i,       ] = True
        for bi in range(BANDS_N):
            all_data[landsati.is_valid,i,bi] = (landsati.reflectance_30m[bi,landsati.is_valid]-x_mean[bi])/x_std[bi] 
        
        ## other informaion 
        doys[i] = landsati.doy
        sensors[i] = sensor_code
    
    ## load dem image
    dem_image = rasterio.open(dem_file).read() # filled value -9999.0
    ## *************************************************************************************************************    
    no_of_obs_image = all_qa.sum(axis=(2))
    if no_of_obs_image.max()>PERIODS:
        print("Note !! level-2 there are some pixels with cloud free observations >PERIODS=80 for this tile no_of_obs_image.max()=" + no_of_obs_image.max())
        # exit() 
    ## *************************************************************************************************************
    ## load model
    model = tf.keras.models.load_model(model_path, compile=False)
    ###################################################################################################################
    ## process when PERIODS>80     
    import transformer_encoder44
    # LONG_TIME = 140
    IMG_WIDTH2=8; N_CLASS=7; layern=3; units=64; head=4; model_drop=0.1; XY_DIM_N=4; PRE_TRAIN=False
    model_long = transformer_encoder44.get_transformer_new_att0_daily_withsensor(n_times=LONG_PERIODS,n_feature=IMG_WIDTH2-2,n_out=N_CLASS,layern=layern, units=units, 
                                                                                 n_head=head, drop=model_drop,is_day_input=True,is_sensor=True, is_sensor_embed=True, is_xy=True,
                                                                                 xy_n=XY_DIM_N, dense_layer_n=PRE_TRAIN)
    
    embedding_name = ""
    for il,ilayer in enumerate(model.layers):
        ilayer1 = model     .layers[il] 
        ilayer2 = model_long.layers[il] 
        name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic!='_']) 
        name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic!='_']) 
        if "embedding" in name_cls:
            embedding_name = ilayer1.name
                    
        if name_cls==name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
            print ("\t"+ilayer.name, end=" ")
            model_long.layers[il].set_weights (model.layers[il].get_weights())
     
    ## *************************************************************************************************************
    ## classification & save image & browse 
    XY_DIM_N = 4
    class_image = np.full([XY_DIM,XY_DIM],fill_value=255,dtype=np.int8)
 
    # valid_pixel_image = no_of_obs_image>0 
    valid_pixel_image = np.logical_and.reduce((no_of_obs_image>0, dem_image[0,:]!=-9999.0, dem_image[1,:]!=-9999.0))
    batchi_index = np.full([XY_DIM,XY_DIM],fill_value=False  ,dtype=bool   )
    batchi_index_sub = np.full([XY_DIM,XY_DIM],fill_value=False  ,dtype=bool   )
    # process line by line 
    column_per_batch = 50 # 1:20:19.268540 bath 32 
    column_per_batch = 200 # has valid >80
    process_data_i = np.full([column_per_batch*XY_DIM,LONG_PERIODS,BANDS_N+2+XY_DIM_N],fill_value=-9999.0,dtype=np.float32)
    process_xy_i   = np.full([column_per_batch*XY_DIM,XY_DIM_N],fill_value=-9999.0,dtype=np.float32)
    # column_per_batch = 500 # exceed gpu
    start = datetime.now()
    print_str = '\n\n\nstart time: '+ str(start) + '\nstation sentinel1_folder #_img vgt_cdl' +'\n======================================'
    print(print_str); 
    use_long_model = False
    for batchi in range(0,XY_DIM,column_per_batch):
        # if batchi>0:
            # break;
        batchi_index[:] = False 
        starti = batchi
        endi = min(column_per_batch+batchi, XY_DIM)
        batchi_index[starti:endi, : ] = True
        process_index = np.logical_and (valid_pixel_image, batchi_index) ## this is 2d
        process_n = process_index.sum()
        if process_n==0:
            print("process " + str(batchi) + " ! process_n = "+str(process_n))
            continue 
        
        ## *******************************************
        ## assign reflectance values to process_data_i
        process_data_i[:] = -9999.0
        use_long_model = False
        if total_n<=PERIODS:
            process_data_i[:process_n,:total_n,:(BANDS_N+2)] = all_data[process_index,:]
            # model_predict = model
        elif total_n<=LONG_PERIODS:
            print("! long model is used total_n<=LONG_PERIODS"  , end="\n")
            process_data_i[:process_n,:total_n,:(BANDS_N+2)] = all_data[process_index,:]
            use_long_model = True 
            # model_predict = model_long
        else:
            no_of_obs_i = all_qa[process_index,:].sum(axis=(0))
            index_gt_80 = no_of_obs_i>0
            if index_gt_80.sum()<=PERIODS:
                process_data_i[:process_n,:index_gt_80.sum(),:(BANDS_N+2)] = all_data[process_index,:,:][:,index_gt_80,:]
            elif index_gt_80.sum()<=LONG_PERIODS:
                print("! long model is used  + str(index_gt_80.sum())", end="\n")
                process_data_i[:process_n,:index_gt_80.sum(),:(BANDS_N+2)] = all_data[process_index,:,:][:,index_gt_80,:]
                use_long_model = True 
            else:
                use_long_model = True 
                process_index = batchi_index ## this is important as all pixels will be processed rather than only >0 observation pixels 
                print("! Warning level-3 cloud free observations >PERIODS=80 for this tile index_gt_80.sum()=" + str(index_gt_80.sum()), end="\n")
                trytime = 0
                success = False
                BREAK_Ns = [2,4,8,10,20,50]
                Break_i = 0
                BREAK_N = BREAK_Ns[Break_i] ## 100 by 100 can solve the problem but not 200*200 images>80
                while (not success):
                    try:
                        # continue break down into images to subsets
                        # BREAK_N = 4 ## for tile h07v13 this is safer on Apr. 21, 2023 
                        for subi in range(0,endi-starti,column_per_batch//BREAK_N):
                            starti_sub = subi+starti
                            endi_sub = min(column_per_batch//BREAK_N+starti_sub, endi)
                            for subj in range(0,XY_DIM,column_per_batch//BREAK_N):
                                startj_sub = subj
                                endj_sub = min(column_per_batch//BREAK_N+subj, XY_DIM)                        
                                batchi_index_sub[:] = False 
                                batchi_index_sub[starti_sub:endi_sub, startj_sub:endj_sub] = True
                                process_index_sub = batchi_index_sub # no need only include valid pixels 
                                process_data_index = process_index_sub[starti:endi, : ].reshape(process_index_sub[starti:endi, : ].size)
                                no_of_obs_i_sub = all_qa[process_index_sub,:].sum(axis=(0))
                                index_gt_80_sub = no_of_obs_i_sub>0 
                                process_data_i[process_data_index,:index_gt_80_sub.sum(),:(BANDS_N+2)] = all_data[process_index_sub,:,:][:,index_gt_80_sub,:]
                    except:
                        print("! ! Error BREAK_N = {:d} cannot fix there are some pixels tile index_gt_80.sum()={:d}".format(BREAK_N,index_gt_80.sum()) )
                        Break_i = Break_i+1;
                        if Break_i>=len(BREAK_Ns):
                            break                        
                        BREAK_N = BREAK_Ns[Break_i] ## 
                        continue                                           
                    success = True                 
                process_n = process_index.sum()
        
        ## *******************************************
        ## assign x, y values to process_xy_i
        print("process " + str(batchi) + " process_n = "+str(process_n))
        process_xy_i [:] = -9999.0
        cols, rows = np.meshgrid(np.arange(starti,endi), np.arange(XY_DIM))
        xys = np.array (rasterio.transform.xy(landsati.profile['transform'], rows, cols) )
        xys = np.moveaxis(xys,2,1)
        process_index_i = process_index[starti:endi, : ]
        if process_index_i.sum()!=process_n:
            print("an error !! process_index_i.sum()!=process_index.sum(): !! ")
        
        # xs, ys = rasterio.transform.xy(landsati.profile['transform'], rows, cols)
        process_xy_i[:process_n,0] = (xys[0,process_index_i].reshape(process_n)-x_mean[BANDS_N+2])/x_std[BANDS_N+2] 
        process_xy_i[:process_n,1] = (xys[1,process_index_i].reshape(process_n)-x_mean[BANDS_N+3])/x_std[BANDS_N+3] 
        process_xy_i[:process_n,2] = (dem_image[0,starti:endi, : ][process_index_i].reshape(process_n)-x_mean[BANDS_N+4])/x_std[BANDS_N+4] 
        process_xy_i[:process_n,3] = (dem_image[1,starti:endi, : ][process_index_i].reshape(process_n)-x_mean[BANDS_N+5])/x_std[BANDS_N+5] 
        
        process_data_i[:process_n,:,(BANDS_N+2):] = process_xy_i[:process_n,np.newaxis,:]
        # logits = model.predict([process_data_i[:process_n,:], process_xy_i[:process_n,:]], batch_size=BATCH)
        if use_long_model==True:
            logits = model_long.predict(process_data_i[:process_n,:], batch_size=BATCH)
        else:
            logits = model.predict(process_data_i[:process_n,:PERIODS], batch_size=BATCH)
        
        class_image[process_index] = np.argmax(logits,axis=1).astype(np.uint8)
        tf.keras.backend.clear_session()
        gc.collect()
        # break 
    
    class_image [np.logical_not(valid_pixel_image) ] = 255
    end = datetime.now()
    elapsed = end-start
    print_str = '\nEnd time = '+str(end) + 'Elapsed time = '+str(elapsed) + '\n======================================'
    print(print_str); 
    
    ## *************************************************************************************************************
    ## save result and browse
    landsati.save_image_file (class_image,prefix="LC_CNN_Daily",folder=CLASS_DIR)
#########################################################################################################################################





