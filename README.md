# CRIT
Classifying the raw irregular time series

Hankui K. Zhang, Dong Luo, Zhongbin Li, Classifying raw irregular Landsat time series (CRIT) for large area land cover mapping by adapting Transformer model 
It contains: 
LCMAP_CU_Landsat_ARD.DAILY.85.06.18.24997.sensor.st.dem.csv   reference data saved in csv file
Pro_lcmap_CRIT_v11_69.py  >> Model training code including load csv file, pre-process data, generate train and test data 
	train_test.py
	transformer_encoder44.py
	customized_train.py
	split folder: the split file that generate the results in the paper
note: the mean and std file will automatically generate during the model training.
Pro_load_model_run_tile_v2_3.py >> major python code to load tile and DEM and predict
	find_files.py
	Landsat_ARD_io.py
