# d:\mycode.RT\mycode.research\Landsat_8_BRDF\
# Hank on Aug 02, 2020 
# Hank on Dec 01, 2021 
# Dump pair data

# import Landsat_ARD_io

## *** ! in csv files note both Landsat 7 and 8 are denoted as band 2, 3, 4, 5, 6 and 7 ! ****
## i.e., follow the Landsat 8 band name convention 
import os
import datetime
import numpy as np
import copy

# import statistics 
# import json
import rasterio

class Landsat_ARD_tile:
    """sentinel scene"""
    ## file names 
    # meta_file = ""
    input_files = ""
    qa_file = ""
    sat_file = ""
    sz_file = ""
    sa_file = ""
    vz_file = ""
    va_file = ""
    
    base_name = ""
    
    band_n = 6
    is_L8 = 0
    
    SIZE_30m = 5000
    # SIZE_10m = 10980
    
    ANGLE_SCALE = 100.0
    profile = 0
    tile_ID = 0
    year = 2019
    doy = 1
    
    # is_srf_nbar = 0 # 0 is srf and 1 is nbar 
    
    ## images 
    is_angle = False
    is_shadow = True
    is_toa = False 
    sz = 0
    sa = 0
    vz = 0
    va = 0
    # *0.0000275-0.2
    MULTI_SCALE = 0.0000275
    ADD_SCALE = -0.2
    
    reflectance_30m = np.empty([1])
    # reflectance_10m = np.empty([1])
    
    ## scl & is_valid are defined at 20 m 
    qa = np.empty([1])
    sat = np.empty([1])
    is_valid = np.empty([1])
    is_fill  = np.empty([1])
    is_cirrusat = np.empty([1])
    is_snow     = np.empty([1])
    
    def __init__(self,input_QA_file, is_L8=1, is_angle=False, is_shadow=True, is_toa=False):
        # self.meta_file = ""
        self.is_L8 = is_L8
        self.is_shadow = is_shadow
        self.qa_file  = copy.copy(input_QA_file)
        self.sat_file = copy.copy(input_QA_file).replace("QA_PIXEL.TIF","QA_RADSAT.TIF")
        self.sz_file  = copy.copy(input_QA_file).replace("QA_PIXEL.TIF","SZA.TIF")
        self.sa_file  = copy.copy(input_QA_file).replace("QA_PIXEL.TIF","SAA.TIF")
        self.vz_file  = copy.copy(input_QA_file).replace("QA_PIXEL.TIF","VZA.TIF")
        self.va_file  = copy.copy(input_QA_file).replace("QA_PIXEL.TIF","VAA.TIF")
        
        self.is_angle = is_angle
        self.is_toa   = is_toa
        keywords = "SR"
        if is_toa:
            keywords = "TOA"
            MULTI_SCALE = 0.0000275
            ADD_SCALE = -0.2
        ## OH my god, they used the same scale for both TOA and surface 
        # self.input_files = np.full([band_n], fill_value=" ")
        self.input_files = list()
        
        if is_L8==1:
            # for i in range(band_n):
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B2.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B3.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B4.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B5.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B6.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B7.TIF"))
        else:
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B1.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B2.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B3.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B4.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B5.TIF"))
            self.input_files.append(copy.copy(input_QA_file).replace("QA_PIXEL.TIF",keywords+"_B7.TIF"))
                
        self.base_name = os.path.basename(self.qa_file)
        self.tile_ID = self.base_name[8:14]
        self.year = int(self.base_name[15:19])
        self.get_doy()
    
    
    def get_doy(self):
        year = int(self.base_name[15:19])
        month = int(self.base_name[19:21])
        day = int(self.base_name[21:23])
        self.doy = (datetime.datetime(year=year, month=month, day=day)-datetime.datetime(year=year, month=1, day=1)).days+1
    
    def normalize_TOA(self):
        self.reflectance_30m[:,self.is_fill==0] = self.reflectance_30m[:,self.is_fill==0]/np.cos(self.sz[self.is_fill==0]/180*np.pi)
            # self.profile = src.profile
        
        
    def load_data(self):
        # self.meta_file = ""
        sz = 0
        sa = 0
        vz = 0
        va = 0
        
        ## 30 m bands 
        # https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-2-science-products
        self.reflectance_30m = np.full([self.band_n, self.SIZE_30m, self.SIZE_30m], fill_value=-9999, dtype=np.float32)
        for i in range(self.band_n):
            if not os.path.exists(self.input_files[i]):
                print ("file not exists "+self.input_files[i])
                return 0
            
            with rasterio.open(self.input_files[i]) as src:
                self.reflectance_30m[i,:,:] = src.read().astype(np.float32)*self.MULTI_SCALE+self.ADD_SCALE
                self.profile = src.profile
        
        ## cloud mask
        ## not work anymore below 
        ## https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1337_Landsat7ETM-C2-L2-DFCB-v5.pdf
        ## https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf
        ## below works on Jul 7 2022
        ## -> https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1337_Landsat7ETM-C2-L2-DFCB-v5.pdf
        ## -> https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf

        if not os.path.exists(self.qa_file):
            print ("file not exists "+self.qa_file)
            return 0
        
        with rasterio.open(self.qa_file) as src:
            self.qa = src.read()[0,:,:]
            self.is_fill = np.bitwise_and(self.qa, 1) # 1 is filled and 0 is image
            # self.is_valid = np.logical_or(self.scl==4, self.scl==5) 
            # self.save_valid_file() # for test only
        
        if not os.path.exists(self.sat_file):
            print ("file not exists "+self.sat_file)
            return 0
        
        with rasterio.open(self.sat_file) as src:
            self.sat = src.read()[0,:,:]
        
        ## whether or not to read angles 
        if self.is_angle:
            ## sz
            if not os.path.exists(self.sz_file):
                print ("file not exists "+self.sz_file)
                return 0
            
            with rasterio.open(self.sz_file) as src:
                self.sz = src.read()[0,:,:].astype(np.float32)/self.ANGLE_SCALE
            
            ## sa
            if not os.path.exists(self.sa_file):
                print ("file not exists "+self.sa_file)
                return 0
            
            with rasterio.open(self.sa_file) as src:
                self.sa = src.read()[0,:,:].astype(np.float32)/self.ANGLE_SCALE
            
            ## vz
            if not os.path.exists(self.vz_file):
                print ("file not exists "+self.vz_file)
                return 0
            
            with rasterio.open(self.vz_file) as src:
                self.vz = src.read()[0,:,:].astype(np.float32)/self.ANGLE_SCALE
            
            ## va
            if not os.path.exists(self.va_file):
                print ("file not exists "+self.va_file)
                return 0
            
            with rasterio.open(self.va_file) as src:
                self.va = src.read()[0,:,:].astype(np.float32)/self.ANGLE_SCALE
        
        # page 25 for L8/9 and page 16 for L5/7 
        # https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1435%20Landsat%20Collection%202%20%28C2%29%20U.S.%20Analysis%20Ready%20Data%20%28ARD%29%20Data%20Format%20Control%20Book%20%28DFCB%29%20-v2.0%202021_02_09.pdf
        if self.is_L8==1:
            if self.is_shadow:
                self.is_valid = np.logical_or.reduce((np.bitwise_and(self.qa,1), np.bitwise_and(np.right_shift(self.qa,1),1) # Fill and cloud edge
                    , np.bitwise_and(np.right_shift(self.qa ,2),1) # Cirrus
                    , np.bitwise_and(np.right_shift(self.qa ,3),1) # Cloud
                    , np.bitwise_and(np.right_shift(self.qa ,4),1) # cloud shadow
                    # , np.bitwise_and(np.right_shift(self.qa ,5),1) # snow 
                    # , np.bitwise_and(np.right_shift(self.qa ,7),1) # water 
                    , np.bitwise_and(np.right_shift(self.sat,0),1)
                    , np.bitwise_and(np.right_shift(self.sat,1),1)
                    , np.bitwise_and(np.right_shift(self.sat,2),1)
                    , np.bitwise_and(np.right_shift(self.sat,3),1)
                    , np.bitwise_and(np.right_shift(self.sat,4),1)
                    , np.bitwise_and(np.right_shift(self.sat,5),1)
                    , np.bitwise_and(np.right_shift(self.sat,6),1)
                    ))
            else:
                self.is_valid = np.logical_or.reduce((np.bitwise_and(self.qa,1), np.bitwise_and(np.right_shift(self.qa,1),1)  # Fill and cloud edge
                    , np.bitwise_and(np.right_shift(self.qa ,2),1) # Cirrus
                    , np.bitwise_and(np.right_shift(self.qa ,3),1) # Cloud
                    # , np.bitwise_and(np.right_shift(self.qa ,4),1) # cloud shadow
                    # , np.bitwise_and(np.right_shift(self.qa ,5),1) # snow 
                    # , np.bitwise_and(np.right_shift(self.qa ,7),1) # water 
                    , np.bitwise_and(np.right_shift(self.sat,0),1)
                    , np.bitwise_and(np.right_shift(self.sat,1),1)
                    , np.bitwise_and(np.right_shift(self.sat,2),1)
                    , np.bitwise_and(np.right_shift(self.sat,3),1)
                    , np.bitwise_and(np.right_shift(self.sat,4),1)
                    , np.bitwise_and(np.right_shift(self.sat,5),1)
                    , np.bitwise_and(np.right_shift(self.sat,6),1)
                    ))
            
            self.is_cirrusat = np.logical_or.reduce((np.bitwise_and(np.right_shift(self.qa ,2),1)
                , np.bitwise_and(np.right_shift(self.sat,0),1)
                , np.bitwise_and(np.right_shift(self.sat,1),1)
                , np.bitwise_and(np.right_shift(self.sat,2),1)
                , np.bitwise_and(np.right_shift(self.sat,3),1)
                , np.bitwise_and(np.right_shift(self.sat,4),1)
                , np.bitwise_and(np.right_shift(self.sat,5),1)
                , np.bitwise_and(np.right_shift(self.sat,6),1)
                ))
            
            # self.is_snow = np.bitwise_and(np.right_shift(self.qa ,5),1) 
        else:
            # This is Landsat 4, 5 & 7
            print ("This is Landsat 4, 5 or 7");
            if self.is_shadow:
                self.is_valid = np.logical_or.reduce((np.bitwise_and(self.qa,1), np.bitwise_and(np.right_shift(self.qa,1),1)  # cloud and cloud edge 
                    # , np.bitwise_and(np.right_shift(self.qa,2),1) # Unused
                    , np.bitwise_and(np.right_shift(self.qa,3),1) # Cloud
                    , np.bitwise_and(np.right_shift(self.qa,4),1) # cloud shadow
                    # , np.bitwise_and(np.right_shift(self.qa,5),1) # snow 
                    # , np.bitwise_and(np.right_shift(self.qa,7),1) # water 
                    , np.bitwise_and(np.right_shift(self.sat,0),1)
                    , np.bitwise_and(np.right_shift(self.sat,1),1)
                    , np.bitwise_and(np.right_shift(self.sat,2),1)
                    , np.bitwise_and(np.right_shift(self.sat,3),1)
                    , np.bitwise_and(np.right_shift(self.sat,4),1)
                    , np.bitwise_and(np.right_shift(self.sat,6),1)
                    ))
            else:
                self.is_valid = np.logical_or.reduce((np.bitwise_and(self.qa,1), np.bitwise_and(np.right_shift(self.qa,1),1)  # cloud and cloud edge 
                    # , np.bitwise_and(np.right_shift(self.qa,2),1)
                    , np.bitwise_and(np.right_shift(self.qa,3),1) # Cloud
                    # , np.bitwise_and(np.right_shift(self.qa,4),1) # cloud shadow
                    # , np.bitwise_and(np.right_shift(self.qa,5),1) # snow 
                    # , np.bitwise_and(np.right_shift(self.qa,7),1) # water 
                    , np.bitwise_and(np.right_shift(self.sat,0),1)
                    , np.bitwise_and(np.right_shift(self.sat,1),1)
                    , np.bitwise_and(np.right_shift(self.sat,2),1)
                    , np.bitwise_and(np.right_shift(self.sat,3),1)
                    , np.bitwise_and(np.right_shift(self.sat,4),1)
                    , np.bitwise_and(np.right_shift(self.sat,6),1)
                    ))
            
        
        self.is_snow = np.bitwise_and(np.right_shift(self.qa ,5),1) 
        # self.is_valid = np.logical_not(self.is_valid) # old one 
        self.is_valid = np.logical_and (np.logical_not(self.is_valid), np.logical_not(self.is_snow)) # fixed on May 17, 2023 to exclude water and include snow 
        return 1
        # scl = np.full([self.SIZE, self.SIZE], fill_value=0, dtype=np.int16)
        # is_valid = np.full([self.SIZE, self.SIZE], fill_value=0, dtype=np.int16)
    
    ## for test only
    def save_valid_file (self):
        naip_meta = self.profile
        naip_meta['dtype'] = 'uint8'
        naip_meta['count'] = 1
        with rasterio.open("./valid."+self.base_name, 'w', **naip_meta) as dst:
            dst.write(self.is_valid.astype(np.uint8), indexes=1 )    
    
    ## for test only
    # image = valid_sum
    def save_image_file (self,image,prefix="total_n",folder="./"):
        with rasterio.open(self.qa_file) as src:
            profile = src.profile        
        
        naip_meta = profile
        naip_meta['tiled'] = False
        naip_meta['compress'] = 'LZW'
        if image.dtype==np.uint8 or image.dtype==np.int8 :
            naip_meta['dtype'] = 'uint8'
        elif image.dtype==np.float32:
            naip_meta['dtype'] = 'float32'
        
        file_name = folder+prefix+self.base_name
        if len(image.shape)==2:
            naip_meta['count'] = 1
            with rasterio.open(file_name, 'w', **naip_meta) as dst:
                dst.write(image, indexes=1 )    
        else: 
            naip_meta['count'] = image.shape[0]
            with rasterio.open(file_name, 'w', **naip_meta) as dst:
                dst.write(image)    
    
    
