dataset_name = 'pollution'   # 'pollution', 'building', 'mimic3'
from preprocess_pollution import *    # Check

model = 'stam_lite'
run_num = 1
print("%s run_num: "%(dataset_name),run_num)
pred_type = 'uni_var_multi_ts'
#inp_var = 9   # Number of input variables
#out_var = 1   # Number of output variables

# fix random seed for reproducibility
from numpy.random import seed 
seed(run_num)
from tensorflow import set_random_seed
set_random_seed(run_num)

import os
import numpy as np
from keras.layers import Concatenate, Dot, Input, LSTM, RepeatVector, Dense
from keras.layers import Dropout, Flatten, Reshape, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras.activations import softmax
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import csv
import pandas as pd
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
# print(os.environ['CUDA_VISIBLE_DEVICES'])
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

h_s = 32  # {16, 32, 64, 128}
dropout = 0.2  # {0.2, 0.5}
batch_size = 256  # {128, 256, 512}
epochs = 5 # 50
lr_rate = 0.001
con_dim = 4   # Reduction in dimension of the temporal context to con_dim before concat with yhat

# Create directory to save results
dir_ = '%s/%s/%s_Tx_%s_Ty_%s_run_%s_outvar_%s_inpvar_%s_hs_%s_dropout_%s_bs_%s_epochs_%s_lr_%s'\
            %(dataset_name, pred_type, model, Tx, Ty, run_num, out_var, inp_var, h_s, dropout, batch_size, epochs, lr_rate)

if not os.path.exists(dir_):
    os.makedirs(dir_)

# Print shapes
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)  # (30652, 10, 8), (30652, 2)
print('x_val shape:', x_val.shape, 'y_val shape:', y_val.shape)     # (8758, 10, 8), (8758, 2)
print('x_test shape:', x_test.shape, 'y_test shape:', y_test.shape)    # (4379, 10, 8), (4379, 2)

# Reshape
y_train = y_train.reshape((y_train.shape[0], Ty, 1))  # (30652, 2, 1)
y_val = y_val.reshape((y_val.shape[0], Ty, 1))        # (8758, 2, 1)
y_test = y_test.reshape((y_test.shape[0], Ty, 1))     # (4379, 2, 1)

# Model
t_repeator = RepeatVector(Tx)
t_densor = Dense(1, activation = "relu")

s_repeator = RepeatVector(inp_var)
s_densor_1 = Dense(h_s, activation = "relu")
s_densor_2 = Dense(1, activation = "relu")   # for attention weights

concatenator = Concatenate(axis=-1)

# Softmax
def softMaxLayer(x):
    return softmax(x, axis=1)   # Use axis = 1 for attention

activator = Activation(softMaxLayer)
dotor = Dot(axes = 1)

decoder_lstm = LSTM(h_s, return_state = True)
flatten = Flatten()

# Temporal Attention
def temporal_attention(a, s_prev):
    
    # s_prev: previous hidden state of decoder (n_samples, 16)
    # a: Sequence of encoder hidden states (n_sample, 10, 16)
    s_prev = t_repeator(s_prev)  # (n_samples, 10, 16)
    concat = concatenator([a, s_prev])   # (n_samples, 10, 32)
    e_temporal = t_densor(concat)  # (n_samples, 10, 1)
    alphas = activator(e_temporal)    # (n_samples, 10, 1)
    t_context = dotor([alphas, a])    # (n_samples, 1, 16)
    
    return t_context, alphas, e_temporal

# Spatial Attention
def spatial_attention(v, s_prev):
    
    # s_prev: previous hidden state of decoder (n_samples, 16)
    # v: variable vectors (n_samples, 8, 10): (n_samples, inp_var, Tx)
    s_fc = s_densor_1(v)      # (n_samples, 8, 16)
    
    s_prev = s_repeator(s_prev)  # (n_samples, 8, 16)
    concat = concatenator([s_fc, s_prev])    # (n_samples, 8, 32)
    e_spatial = s_densor_2(concat)  # (n_samples, 8, 1)
    betas = activator(e_spatial) # (n_samples, 8, 1)
    s_context = dotor([betas, s_fc])  # (n_samples, 1, 16)
    
    return s_context, betas, e_spatial

# Model
def model(Tx, Ty, inp_var, h_s, dropout):
    
    # Tx : Number of input timesteps
    # Ty : Number of output timesteps
    # inp_var: Number of input variables
    # h_s: Hidden State Dimensions for Encoder, Decoder
    encoder_input = Input(shape = (Tx, inp_var))   # (None, 10, 8)
    spatial_input = Input(shape = (inp_var, Tx))    # (None, 8, 10)
    
    # Initialize
    s0 = Input(shape=(h_s,))  # Initialize hidden state for decoder   (None, 16)
    c0 = Input(shape=(h_s,))  # Initialize cell state for decoder     (None, 16)
    yhat0 = Input(shape=(1, ))  # Initialize prev pred y   (None, 1)
    s = s0
    c = c0
    yhat = yhat0  # For regression its a scalar, not a one-hot vector like in NLP
    
    # Lists to store outputs
    outputs = list()
    alphas_betas_list = list()
    
    # Encoder LSTM, Pre-attention        
    lstm_1, state_h, state_c = LSTM(h_s, return_state=True, return_sequences=True)(encoder_input)
    lstm_1 = Dropout (dropout)(lstm_1)     # (None, 10, 16)
    
    lstm_2, state_h, state_c = LSTM(h_s, return_state=True, return_sequences=True)(lstm_1)
    lstm_2 = Dropout (dropout)(lstm_2)     # (None, 10, 16)

    # Decode for Ty steps
    for t in range(Ty):
        
        # Temporal Attention
        t_context, alphas, e_temporal = temporal_attention (lstm_2, s)  # (None, 1, 16)
        
        # Spatial Attention
        s_context, betas, e_spatial = spatial_attention (spatial_input, s)  # (None, 1, 16)
    
        context = concatenator([t_context, s_context])   # (None, 1, 32)
    
        context = Dense (con_dim, activation = "relu")(context)  # (None, 1, 4)
        context = flatten(context)  # (None, 4)
        context = concatenator([context, yhat])   # (None, 5)
        context = Reshape((1, con_dim + 1))(context)   # (None, 1, 5)
        
        # Decoder LSTM
        s, _, c = decoder_lstm(context, initial_state=[s, c])
        s = Dropout (dropout)(s)   # (None, 16)
        
        # FC Layer
        yhat = Dense (1, activation = "linear")(s)
        
        # Append lists
        outputs.append(yhat)
        
        # Append lists
        alphas_betas_list.append(alphas)
        alphas_betas_list.append(betas)
        alphas_betas_list.append(yhat)
        
    pred_model = Model([encoder_input, spatial_input, s0, c0, yhat0], outputs)   # Prediction Model
    prob_model = Model([encoder_input, spatial_input, s0, c0, yhat0], alphas_betas_list)    # Weights Model
        
    return pred_model, prob_model

# Model Summary
pred_model, prob_model = model(Tx, Ty, inp_var, h_s, dropout)
pred_model.summary()

# Train Model
s0_train = np.zeros((y_train.shape[0], h_s))
c0_train = np.zeros((y_train.shape[0], h_s))
yhat0_train = np.zeros((y_train.shape[0], 1))

s0_val = np.zeros((y_val.shape[0], h_s))
c0_val = np.zeros((y_val.shape[0], h_s))
yhat0_val = np.zeros((y_val.shape[0], 1))

# Transpose of timesteps and variables
s_train = x_train.transpose(0, 2, 1)    # (30652, 8, 10)   Transpose of axes 2 and 3 keeping axis 1 same
s_val = x_val.transpose(0, 2, 1)        # (8758, 8, 10)    Transpose of axes 2 and 3 keeping axis 1 same

#outputs = list(Yoh.swapaxes(0,1)) 
outputs_train = list(y_train.swapaxes(0,1))    # Ty numpy lists each (30562, 1)
outputs_val = list(y_val.swapaxes(0,1))        # Ty numpy lists each (8758, 1)

pred_model.compile(loss='mean_squared_error', optimizer = Adam(lr=lr_rate)) 
start_time=time()
hist = pred_model.fit ([x_train, s_train, s0_train, c0_train, yhat0_train], outputs_train,
                  batch_size = batch_size,
                  epochs = epochs,
                  #callbacks = callback_lists,   # Try Early Stopping
                  verbose = 2,
                  shuffle = True,
                  validation_data=([x_val, s_val, s0_val, c0_val, yhat0_val], outputs_val))
print("Per epoch train time:",(time()-start_time)/epochs)

# Attention Weights Model
prob_model.set_weights(pred_model.get_weights())

# Plot
loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.figure()
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.savefig('%s/loss_plot.png'%(dir_))
print("Saved loss plot to disk") 
plt.close()

# Save Data
loss = pd.DataFrame(loss).to_csv('%s/loss.csv'%(dir_))    # Not in original scale 
val_loss = pd.DataFrame(val_loss).to_csv('%s/val_loss.csv'%(dir_))  # Not in original scale

# MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Plot Ground Truth, Model Prediction
def actual_pred_plot (y_actual, y_pred, n_samples = 60):
    
    # Shape of y_actual, y_pred: (8758, Ty)
    plt.figure()
    plt.plot(y_actual[ : n_samples, -1])  # 60 examples, last prediction time step
    plt.plot(y_pred[ : n_samples, -1])    # 60 examples, last prediction time step
    plt.legend(['Ground Truth', 'Model Prediction'], loc='upper right')
    plt.savefig('%s/actual_pred_plot.png'%(dir_))
    print("Saved actual vs pred plot to disk")
    plt.close()

# Correlation Scatter Plot
def scatter_plot (y_actual, y_pred):
    
    # Shape of y_actual, y_pred: (8758, Ty)
    plt.figure()
    plt.scatter(y_actual[:, -1], y_pred[:, -1])
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=4)
    plt.title('Predicted Value Vs Actual Value')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.savefig('%s/scatter_plot.png'%(dir_))
    print("Saved scatter plot to disk")
    plt.close()
           
# Evaluate Model
def evaluate_model (x_data, y_data, dataset):
    
    # x_train: (30652, 10, 8), y_train: (30652, 2, 1)
    s_data = x_data.transpose(0, 2, 1)   # (30652, 8, 10)
    
    s0_data = np.zeros((y_data.shape[0], h_s))
    c0_data = np.zeros((y_data.shape[0], h_s))
    yhat0_data = np.zeros((y_data.shape[0], 1))
    if dataset== "test":
        start_time=time()        
    y_data_hat = pred_model.predict([x_data, s_data, s0_data, c0_data, yhat0_data], batch_size = batch_size)
    if dataset== "test":
        print("Total testing time: ", time()-start_time)   
    y_data_hat = np.array(y_data_hat)     # (2, 30652, 1)
    
    if Ty != 1:     # For Ty = 1, y_data_hat shape is already (30652, 1)
        y_data_hat = y_data_hat.swapaxes(0,1)     # (30652, 2, 1)

    y_data_hat = y_data_hat.reshape((y_data_hat.shape[0], Ty))    # (30652, 2)
    y_data_hat = scaler_y.inverse_transform(y_data_hat)
    
    y_data = y_data.reshape((y_data.shape[0], Ty))      # (30652, 2)
    y_data = scaler_y.inverse_transform(y_data)
    
    # Selecting the output only for Ty timestep
    y_data_hat_Ty = y_data_hat [:, (Ty - 1)]   # (30652, )
    y_data_Ty = y_data [:, (Ty - 1)]    # (30652, )
    
    metric_dict = {}  # Dictionary to save the metrics
    
    data_rmse = sqrt(mean_squared_error(y_data_Ty, y_data_hat_Ty))
    metric_dict ['rmse'] = data_rmse 
    print('%s RMSE: %.4f' %(dataset, data_rmse))
    
    data_mae = mean_absolute_error(y_data_Ty, y_data_hat_Ty)
    metric_dict ['mae'] = data_mae
    print('%s MAE: %.4f' %(dataset, data_mae))
    
    data_r2score = r2_score(y_data_Ty, y_data_hat_Ty)
    metric_dict ['r2_score'] = data_r2score
    print('%s r2_score: %.4f' %(dataset, data_r2score))
            
    # Save metrics
    with open('%s/metrics_%s.csv' %(dir_, dataset), 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in metric_dict.items():
            writer.writerow([key, value])
    
    # Save Actual Vs Predicted Plot and Scatter PLot for test set
    if dataset == 'test':
        actual_pred_plot (y_data, y_data_hat)
        scatter_plot (y_data, y_data_hat)
        
    return y_data_Ty, y_data_hat_Ty, metric_dict

# Get numpy array of alphas in shape (30652, 10, 2) from list of 2(Ty) elements each element (30652, 10, 1)
def alphas_array(prob_list):
    
    prob = np.array(prob_list)
    prob = prob.swapaxes(0, 1)
    prob = prob.swapaxes(1, 2)
    prob = prob.reshape((prob.shape[0], prob.shape[1], prob.shape[2]))
    
    return prob

# Get Attention Weights
def get_weights (x_data, y_data, dataset):
    
    s_data = x_data.transpose(0, 2, 1)   # (30652, 8, 10)
    
    s0_data = np.zeros((y_data.shape[0], h_s))
    c0_data = np.zeros((y_data.shape[0], h_s))
    yhat0_data = np.zeros((y_data.shape[0], 1))
    
    y_data_hat_prob = prob_model.predict([x_data, s_data, s0_data, c0_data, yhat0_data], batch_size = batch_size)
    len_list = len(y_data_hat_prob)   # List of 6 elements  
    
    # alphas list: elements 0, 3, each element (30652, 10, 1)
    y_data_hat_alphas = [y_data_hat_prob[i] for i in range(0, len_list, 3)]
    y_data_hat_alphas = alphas_array (y_data_hat_alphas)      # (30652, 10, 2)
    np.save("%s/y_%s_hat_alphas"%(dir_, dataset), y_data_hat_alphas)  # y_val_hat_alphas
    
    # betas list: elements 1, 4, each element (30652, 8, 1)
    y_data_hat_betas = [y_data_hat_prob[i] for i in range(1, len_list, 3)]
    y_data_hat_betas = alphas_array (y_data_hat_betas)     # (30652, 8, 2)
    np.save("%s/y_%s_hat_betas"%(dir_, dataset), y_data_hat_betas)  # y_val_hat_betas
    
    return y_data_hat_alphas, y_data_hat_betas

# Evaluate Model - Train, Validation, Test Sets
y_train_Ty, y_train_hat_Ty, train_metrics = evaluate_model (x_train, y_train, 'train')
y_val_Ty, y_val_hat_Ty, val_metrics = evaluate_model (x_val, y_val, 'val')
y_test, y_test_hat_Ty, test_metrics = evaluate_model (x_test, y_test, 'test')

# Attention Weights - Train, Validation, Test Sets
y_train_hat_alphas, y_train_hat_betas = get_weights(x_train, y_train, 'train')
y_val_hat_alphas, y_val_hat_betas = get_weights(x_val, y_val, 'val')
y_test_hat_alphas, y_test_hat_betas = get_weights(x_test, y_test, 'test')

# Average of attention weights across all samples
y_train_hat_betas_avg = np.mean(y_train_hat_betas, axis = 0)
y_val_hat_betas_avg = np.mean(y_val_hat_betas, axis = 0)
y_test_hat_betas_avg = np.mean(y_test_hat_betas, axis = 0)