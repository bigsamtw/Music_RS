#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tfcf.metrics import mae
from tfcf.metrics import rmse
from tfcf.datasets import ml1m
from tfcf.config import Config
from tfcf.models.svd import SVD
# from tfcf.models.svd import SVDPP
from sklearn.model_selection import train_test_split

dir_ = '../data/'
file_name = 'normalized_filter_track_5_user_100.csv'


# In[11]:


# Note that x is a 2D numpy array, 
# x[i, :] contains the user-item pair, and y[i] is the corresponding rating.

df = pd.read_pickle(os.path.join(dir_, file_name[:-3] + 'pkl'))

x_train = np.loadtxt(os.path.join(dir_, 'train_x_' + file_name), delimiter=',')
x_test  = np.loadtxt(os.path.join(dir_, 'test_x_' + file_name), delimiter=',')
y_train = np.loadtxt(os.path.join(dir_, 'train_y_' + file_name), delimiter=',')
y_test  = np.loadtxt(os.path.join(dir_, 'test_y_' + file_name), delimiter=',')    


# In[17]:


config = Config()
config.num_users = len(df['uid'].unique())
config.num_items = len(df['tid'].unique())
config.min_value = df['rating'].min()
config.max_value = df['rating'].max()


# In[ ]:


with tf.Session() as sess:
    # For SVD++ algorithm, if `dual` is True, then the dual term of items' 
    # implicit feedback will be added into the original SVD++ algorithm.
    # model = SVDPP(config, sess, dual=False)
    # model = SVDPP(config, sess, dual=True)
    model = SVD(config, sess)
    model.train(x_train, y_train, validation_data=(
        x_test, y_test), epochs=20, batch_size=1024)
        
    y_pred = model.predict(x_test)
    print('rmse: {}, mae: {}'.format(rmse(y_test, y_pred), mae(y_test, y_pred)))
        
    # Save model
    model = model.save_model('model/')
    
    # Load model
    # model = model.load_model('model/')


np.savetxt(os.path.join(dir_, 'prediction_svd_' + file_name), y_pred, delimiter=",")
