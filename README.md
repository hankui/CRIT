# CRIT
Classifying the raw irregular time series (CRIT) codes using Transformer for the paper: 
Zhang, H. K., Luo, D., & Li, Z. (2024). Classifying raw irregular time series (CRIT) for large area land cover mapping by adapting transformer model. Science of Remote Sensing, 100123.
https://doi.org/10.1016/j.srs.2024.100123

The reference training and testing data saved in csv can be found in https://zenodo.org/records/8097697 (LCMAP_CU_Landsat_ARD.DAILY.85.06.18.24997.sensor.st.dem.csv)

Pro_lcmap_CRIT_v11_69.py  >> CRIT codes including loading csv file, pre-processing data, splitting train and test data, CRIT training and evaluation, and saving trained model
	train_test.py
	transformer_encoder44.py
	customized_train.py
	split folder: the split file use to define training and testing split that generate the results in Zhang et al. (2024)

Pro_load_model_run_tile_v2_3.py >> CRIT codes to load ARD tile, DEM and trained models for land cover mapping
	find_files.py
	Landsat_ARD_io.py

Other files used in comparison experiments in Zhang et al. (2024): 
	v11_694 1D-CNN to classify daily raw irregular time series with DEM/xy 
	v9_31 CRIT without DEM/xy and v9_34 1D CNN for daily raw irregular time series without DEM/xy 
	v9_21 Transformer for 16-day composites and v9_24 1D CNN for 16-day composites 
	v9_52 Transformer for percentiles and v9_53 1D CNN for percentiles
