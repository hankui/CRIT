# import customized_train
import math 
import numpy as np 
import logging

import tensorflow as tf 
import train_test
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from keras.callbacks import TensorBoard
from typing import Callable, Union

# @tf.function
# def loss_reg(model, x, y, training=True):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # y_ = model(x, training=training)
    # y_ = tf.reshape(model(x, training=training),y.shape ) ## a bug on Dec 29 2020 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # mse = tf.keras.losses.MSE(y_true=y, y_pred=y_)
    # return tf.math.reduce_sum(mse)
    # return tf.math.sqrt(tf.math.reduce_mean(mse)) ## changed on Dec 31 2020 & fixed 

# y_pred=model2.predict(X_rnn_train[:16,:,:])
# y_true=y_rnn_train[:16,:,:]
# maskValue=-999
def mask_loss_fixed(maskValue=-9999.0):
    def mask_loss_in(y_true,y_pred):
        isMask = K.equal(y_true, maskValue) #true for all mask values
        isMask = K.cast(isMask, dtype=K.floatx())
        isMask = 1 - isMask #now mask values are zero, and others are 1
        y_true = y_true * isMask   
        y_pred = y_pred * isMask 
        axis_to_reduce = range(1, K.ndim(y_true))       
        # axis_to_reduce = range(0, K.ndim(y_true))       
        sum_square = K.sum(K.square(y_true-y_pred), axis=axis_to_reduce)
        sum_n = K.sum(isMask, axis=axis_to_reduce)
        # loss1 = sum_square/sum_n
        loss1 = K.sqrt(sum_square/sum_n)
        # loss2 = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        return loss1
    return mask_loss_in

## **********************************************************************************************************************************************
## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, target_lr=1e-3, start_lr=0.0):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step_tf = tf.cast(global_step, tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(tf.constant(np.pi) * (global_step_tf - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
    
    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)
    
    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold, learning_rate, target_lr)
    
    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return tf.cast(learning_rate, tf.float32)

## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold
    
    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=step, total_steps=self.total_steps, warmup_steps=self.warmup_steps, start_lr=self.start_lr, 
            target_lr=self.target_lr, hold=self.hold)
        
        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")
    
    def get_config(self):
        config = {
                  'start_lr':self.start_lr,
                  'target_lr':self.target_lr,
                  'warmup_steps':self.warmup_steps,
                  'total_steps':self.total_steps,
                  'hold':self.hold}
        return config

# from keras import backend as K

# https://github.com/tensorflow/tensorflow/issues/39782
class LRTensorBoard(TensorBoard):
    # add other arguments to __init__ if you need
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        optimizer = self.model.optimizer
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        
        logs.update({'lr': K.eval(current_lr)})
        super().on_epoch_end(epoch, logs)
 
def my_train_1schedule(model,train_datasetx,train_datasety,epochs=10,start_rate=0.001,loss=np.nan,per_epoch=100,split_epoch=4,option=0,decay=1e-5, 
        batch_size=32,validation_split=0.04,hold_epoch=0,reduce_epoch=False):
    momentum = 0.9   # good for RMSprop not good for RMSprop but OK for Adam
    # decay = 0.5
    # decay = 0.05
    # decay = 1e-5
    print_str = "momentum="+str(momentum) + "\tlearning rate="+str(start_rate)+"  decay="+str(decay) 
    print(print_str); logging.info(print_str)
    ##*****************************************************************************************************
    ## warm up
    # warm_up = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=start_rate/split_epoch/per_epoch,decay_steps=split_epoch*per_epoch,end_learning_rate=start_rate,name='Decay_linear')
    total_steps = per_epoch*epochs
    warmup_steps = per_epoch*split_epoch
    hold_steps = per_epoch*hold_epoch
    schedule_one = WarmUpCosineDecay(start_lr=0.0, target_lr=start_rate, warmup_steps=warmup_steps, total_steps=total_steps, hold=hold_steps)
    
    if option==0:
        print ('tfa.optimizers.RMSprop '); optimizer=tf.keras.optimizers.RMSprop(learning_rate=schedule_one, rho=momentum    ); 
    elif option==1:                                                                           
        print ('tfa.optimizers.Adam ');   optimizer=tf.keras.optimizers.Adam   (learning_rate=schedule_one, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        print ('tfa.optimizers.AdamW ');  optimizer=tfa.optimizers.AdamW       (weight_decay=decay,learning_rate=schedule_one, beta_1=momentum)

    # metric = get_lr_metric(optimizer)
    # lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=split_epoch, verbose=2)
    if reduce_epoch:
        final_epochs = (epochs-10) if epochs>10 else epochs
    else:
        final_epochs = epochs
    
    if validation_split>0:
        model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=0.04, epochs=final_epochs, verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
    else:
        model_history = model.fit(x=train_datasetx, y=train_datasety, epochs=final_epochs, verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])    
    
    print (model_history.history['lr'])
        
    ##*****************************************************************************************************
    ## consine
    # cos_dec1 = tf.keras.optimizers.schedules.CosineDecay(start_rate, decay_steps=(epochs-split_epoch)*per_epoch, alpha=0, name='Cosine_Decay_1')
    # if option==0:
        # print ('tfa.optimizers.RMSprop'); optimizer=tf.keras.optimizers.RMSprop(learning_rate=cos_dec1, rho=momentum    ); 
    # elif option==1:                                                                           
        # print ('tfa.optimizers.Adam ') ; optimizer=tf.keras.optimizers.Adam   (learning_rate=cos_dec1, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    # else:
        # print ('tfa.optimizers.AdamW '); optimizer=tfa.optimizers.AdamW       (weight_decay=decay,learning_rate=cos_dec1, beta_1=momentum)
    
    # lr_metric = get_lr_metric(optimizer)
    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy',lr_metric])
    # if validation_split>0:
        # model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=0.04, epochs=epochs-split_epoch, verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
    # else:
        # model_history = model.fit(x=train_datasetx, y=train_datasety, epochs=epochs-split_epoch, verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
    
    # print (model_history.history['lr'])
    return model_history
 
## **********************************************************************************************************************************************
## testing data accuacy 
def test_accuacy(model,input_images,y_test):
    logits = model.predict(input_images)
    prop = tf.nn.softmax(logits).numpy() # added on Mar 15 2023 
    classesi = np.argmax(prop,axis=1).astype(np.uint8).reshape(y_test.shape)
    accuracy = (y_test==classesi).sum()/classesi.size
    return accuracy,classesi
