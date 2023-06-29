# Dec 29, 2020
# model builder
# import transformer_encoder44

# refer to
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/quickstart/advanced

import math
import numpy as np
import logging
import pandas as pd

import tensorflow as tf
from keras import backend as K

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization,LayerNormalization
from tensorflow.keras.layers import LSTM,GRU,Bidirectional

from tensorflow.keras.layers import Dropout,Softmax
from tensorflow.keras.layers import LayerNormalization, MaxPooling1D, AveragePooling1D,Conv1D
from tensorflow.keras.layers import Masking,Embedding
from tensorflow.keras.layers import SimpleRNN, Attention, AdditiveAttention, TimeDistributed, MultiHeadAttention
from tensorflow.keras import Input, Model
# import keras_layers.layers as customized 

N_times = 14
N_feature = 14
N_outputs = 8

##************************************************************************************************************
## point_wise_feed_forward_network
def point_wise_feed_forward_network(d_model, dff, reg=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=reg),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_regularizer=reg)  # (batch_size, seq_len, d_model)
    ])

##************************************************************************************************************
## check test_mask.py to see why this is like adding two new axises
def create_padding_mask(inputs, mask_value=0):
    seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

## check test_mask.py to see why this is like adding two new axises
def create_padding_mask_any(inputs0, mask_value=0):
    # seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    if np.isnan(mask_value):
        seq = tf.cast(tf.math.reduce_any(tf.math.logical_not(tf.math.is_nan(inputs0)),axis=2), tf.float32)
    else:
        seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs0, mask_value),axis=2), tf.float32)
    
    return seq[:, tf.newaxis, tf.newaxis, :],seq[:, :, tf.newaxis]  # (batch_size, 1, 1, seq_len)


MAX_L_IN16DAY = 4

# *****************************************************************************************************************************************************
# ****************************** input is daily and 3D  ***************************************************************
# *****************************************************************************************************************************************************
# layern=3; units=64; n_times=80; n_feature=6; n_head=4; is_batch=True; drop=0.1; mask_value = -9999.0;  active="exponential"; n_out=7
# L2=0; is_day_input=True; is_sensor=True; is_sensor_embed=True; is_xy=True 
# inputs = input_images_train_norm3[:2,:,:] # no filled data
# https://www.tensorflow.org/text/tutorials/transformer
def get_transformer_new_att0_daily_withsensor(n_times=14, n_feature=2, n_out=9, layern=3, units=128, n_head=4, drop=0.1, is_batch=True, mask_value=-9999.0, active="softmax", 
        L2=0,is_day_input=False, is_sensor=False, is_sensor_embed=False, is_xy=False, xy_n=2, is_reflectance=False, dense_layer_n=0,newunits=128):
    """
    transformer model considering doy and sensor by adding postional coding
    using AveragePooling1D with mask
    n_times: e.g. 80  > total 80 records
    n_features: 12 in this case
    n_out: classes (e.g. 7 classes)
    parameters: layern=3, units=128, n_head=4, drop=0.1, is_batch=True, active="softmax", L2=0
    mask value: filled data. has been set as -9999.0
    additional informaiton to control data (doy, sensor, etc.):is_day_input=False, is_sensor=False, is_sensor_embed=False, is_xy=False, xy_n=2, is_reflectance=False
    dense_layer_n=0,newunits=128 ()when dense_layer_n >0 need newunits
    """
    inputs = Input(shape=(n_times, n_feature+1+is_sensor+is_xy*xy_n,))      
    # inputs = Input(shape=(n_times, n_feature+1+is_sensor+is_xy,))     #   changed on Apr 19 2023  
    reg = None
    if L2>0:
        reg = tf.keras.regularizers.l2(l=L2)
    
    ## *******************
    # reflectance
    x0 = inputs[:,:,:n_feature]
    mask_multi0 = tf.cast(tf.math.not_equal(x0,mask_value), tf.float32)
    x0 = x0 * mask_multi0
    
    ## *******************
    # positional -> need to change positional to day of year
    if is_day_input==True: ## day of year as position 
        xpp = inputs[:,:,n_feature:(n_feature+1)]
        mask_multi2 = tf.cast(tf.math.not_equal(xpp,mask_value), tf.float32)
        doy_array = np.array(range(1,366)) 
        xpp = (xpp-doy_array.mean() )/doy_array.std ()* mask_multi2
    elif is_day_input==2:  ## cosine and sine positions 
        print ("2d time encoder ")
        xpp = inputs[:,:,n_feature:(n_feature+1)]
        mask_multi2 = tf.cast(tf.math.not_equal(xpp,mask_value), tf.float32)
        # aa = np.array(range(n_times))
        # doy_array = np.array(range(1,366))
        # a = np.sin(doy_array/366*np.pi)
        # a = np.cos(doy_array/366*np.pi)
        xp1 = tf.math.cos(xpp/366*np.pi)/0.7061374773982056
        xpp1 = K.ones_like(inputs[:, :, :1]) * xp1 
        xp2  = (tf.math.sin(xpp/366*np.pi)-0.638360016676796)/0.30637616115177674
        xpp = tf.concat ([xp1,xp2],axis=2) * mask_multi2
    else:  ## position implied in data position 
        xp = np.arange(n_times)[:, np.newaxis] / n_times  # (seq, 1)
        xpp = K.ones_like(inputs[:, :, :1]) * xp        
    
    ## *******************
    # sensor
    if is_sensor:
        xse = inputs[:,:,(n_feature+1):(n_feature+2)]
        xse = xse * mask_multi2

    ## *******************
    # lat lon 
    if is_xy:
        # xxy = inputs[:,:,(n_feature+2):(n_feature+4)]
        xxy = inputs[:,:,(n_feature+2):(n_feature+2+xy_n)]
        xxy = xxy * mask_multi2
    
    ## *******************
    ## embedding inputs 
    embedding_x = Dense(units, use_bias=False)
    # embedding_x = Dense(units) # changed on Apr 19 2023 
    embedding_p = Dense(units, use_bias=False)
    embedding_s = Dense(units, use_bias=False)
    embedding_s_em = Embedding(4, units)
    # embedding_xy = embedding_lon_lat(layer_n=3, units=units, drop=0, reg=reg)
    embedding_xy = Dense(units, use_bias=False)
    if is_sensor and is_sensor_embed and is_xy:
        print ("Use Embedding in sensor encoder and use xy encoder ")
        xse_encoder = embedding_s_em(xse)
        x = embedding_x(x0) + embedding_p(xpp) + xse_encoder[:,:,0,:] + embedding_xy(xxy)
    elif is_sensor and is_sensor_embed:
        print ("Use Embedding in sensor encoder")
        xse_encoder = embedding_s_em(xse)
        x = embedding_x(x0) + embedding_p(xpp) + xse_encoder[:,:,0,:]
    elif is_sensor:
        print ("Use Dense in sensor encoder")
        x = embedding_x(x0) + embedding_p(xpp) + embedding_s(xse)
    else:
        x = embedding_x(x0) + embedding_p(xpp)
    
    ## *******************
    ## start to encoder and decoder    
    # padding_mask, padding_mask3d = create_padding_mask_any(inputs0=inputs[:,:,:], mask_value=mask_value)
    padding_mask, padding_mask3d = create_padding_mask_any(inputs0=inputs[:,:,:6], mask_value=mask_value) # fix this bug on Feb 18 2023, mask only applied for 6 bands but not for all data
    # encoder
    for i in range(layern):
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head, kernel_regularizer=reg)(query=x, value=x, key=x,
            return_attention_scores=True, attention_mask=padding_mask)
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        
        ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg)(out1)
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        
        x = out2
    
    ## *******************
    ## start to output 
    if is_reflectance:
        output = Dense(6, activation="sigmoid", kernel_regularizer=reg)(x)
    else:
        # newunits = units//2 
        # if newunits!=units: # v10.77
            # x = Dense(newunits, kernel_regularizer=reg)(x) # fix this bug on Mar 14, 2023
            # if drop > 0:
                # x = Dropout(drop)(x)

        # x = Dense(newunits, kernel_regularizer=reg)(x) # fix this bug on Mar 14, 2023  # v10.78/79
        for i in range(dense_layer_n):
            x = Dense(newunits, kernel_regularizer=reg, activation='relu')(x) # fix this bug on Mar 14, 2023  # v10.90
            if drop > 0:
                x = Dropout(drop)(x)
        
        if dense_layer_n>0:
            attn_output, attn4 = MultiHeadAttention(key_dim=newunits // n_head, num_heads=n_head, kernel_regularizer=reg)(query=x, value=x, key=x,
                return_attention_scores=True, attention_mask=padding_mask)
            if drop > 0:
                attn_output = Dropout(drop)(attn_output)
            
            out1 = x + attn_output
            if is_batch == True:
                out1 = LayerNormalization(epsilon=1e-6)(out1)
            
            ffn_output = point_wise_feed_forward_network(newunits, newunits * 4, reg=reg)(out1)
            if drop > 0:
                ffn_output = Dropout(drop)(ffn_output)
            
            out2 = out1 + ffn_output
            if is_batch == True:
                out2 = LayerNormalization(epsilon=1e-6)(out2)
            
            x = out2    
        
        ## average
        enc_output2 = tf.math.multiply(x,padding_mask3d)
        x = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(padding_mask3d, axis=1))   
        # x = enc_output
        # for i in range(dense_layer_n):
            # x = Dense(units,activation='relu') (x) 
            # if drop > 0:
                # x = Dropout(drop)(x)
            # if is_batch == True:
                # x = LayerNormalization(epsilon=1e-6)(x)
        
        # output = Dense(n_out, activation=active, kernel_regularizer=reg)(enc_output)
        output = Dense(n_out, kernel_regularizer=reg)(x) # fix this bug on Mar 14, 2023
    
    model = Model(inputs, output)    
    return model



def positional_encoding(length, depth):
    depth = depth // 2
    
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


