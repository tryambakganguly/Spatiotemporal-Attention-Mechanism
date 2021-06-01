# fix random seed for reproducibility
from numpy.random import seed 
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

from pandas import DataFrame, concat, read_csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

Tx = 5  # {5, 10, 12, 18, 24}
Ty = 1   # {1, 2, 3, 4}

inp_var = 8   # Number of input variables
out_var = 1   # Number of output variables
train_split = 0.6   # Size of the training set 
val_split = 0.2    # Size of the validation set
data_file = 'data/pollution.csv'

# Convert series to supervised learning setup
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# load dataset
dataset = read_csv(data_file, header=0, index_col=0)
values = dataset.values

# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])

# ensure all data is float
values = values.astype('float32')      # (43800, 8)

# Column Wise (8 variables)
# Pollution
# Dew
# Temperature 
# Pressure
# Wind Direction
# Wind Speed 
# Snow
# Rain

# frame as supervised learning
reframed = series_to_supervised(values, Tx, Ty)
reframed = reframed.values     # (43789, 96)

n_obs = Tx * inp_var   # 80

X = reframed[:, : n_obs]    # (43789, 80)
y1 = reframed [:, n_obs :]  # (43789, 16)

y = np.zeros ((y1.shape[0], Ty))  # (43789, 2)

# Select only pollution
for i in range(0, Ty):
    index = inp_var * i    # 0, 8, 16, 24, 32
    y [:, i] = y1 [:, index]

# Input Data
X = X.reshape((X.shape[0], Tx, inp_var))  # (43789, 10, 8)
    
# Function for train, test, validation sets with sequential split
def seq_split (x_data, y_data, train_split, val_split):
    
    train_size = int(len(X) * train_split)
    val_size =  int(len(X) * (train_split + val_split))
    
    x_train = X [0 : train_size, :, :]
    y_train = y [0 : train_size]
    
    x_val = X [train_size : val_size, :, :]
    y_val = y [train_size : val_size]
    
    x_test = X [val_size : len(X), :, :]
    y_test = y [val_size : len(y), :]
    
    return x_train, y_train, x_val, y_val, x_test, y_test
    
# train, validation, test sets
x_train, y_train, x_val, y_val, x_test, y_test = seq_split (X, y, train_split, val_split)

# Scale features 
# Two separate scalers for X, Y (diff dimensions) 
scaler_x = MinMaxScaler(feature_range=(-1, 1))
scaler_y =  MinMaxScaler(feature_range=(-1, 1))

x_train_reshaped = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

# Scaling Coefficients calculated from the training dataset
scaler_x = scaler_x.fit(x_train_reshaped)   
scaler_y = scaler_y.fit(y_train)   # (30650, 4)

# Function to scale features after fitting
def scale_features (x_data, y_data):
    
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1] * x_data.shape[2]))
    x_data = scaler_x.transform(x_data)
    x_data = x_data.reshape((x_data.shape[0], Tx, inp_var))
    
    y_data = scaler_y.transform(y_data)
    
    return x_data, y_data

# Scale features
x_train, y_train = scale_features (x_train, y_train)     # (30652, 10, 8), (30652, 2)
x_val, y_val = scale_features (x_val, y_val)    # (8758, 10, 8), (8758, 2)
x_test, y_test = scale_features (x_test, y_test)     # (4379, 10, 8), (4379, 2)
