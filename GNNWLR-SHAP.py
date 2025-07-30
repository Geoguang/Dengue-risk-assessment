import numpy as np
from osgeo import gdal
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
import random
os.chdir('/sh3/ysuanbase/home/yeesuan14577/yeesuan/test/')

####  Core GNNWLR
FILE_PATH = "CV/dengue.csv"
PRECESS_MODE = 0 
RANDOM_STATE = 0 

data = pd.read_csv(FILE_PATH)  
data_coords_x = np.array(data['Longitude'], dtype=np.float64)  
data_coords_y = np.array(data['Latitude'], dtype=np.float64)  
data_coords = list(zip(data['Longitude'], data['Latitude']))    
data_coords = np.array(data_coords, dtype=np.float64)     

name_x = ['bio1','bio5','bio6', 'bio12','bio13','bio14','ISA','DEM','POP','albopictus','aegypti']   
name_y = 'label'   
name_all = name_x.copy()   
name_all.append(name_y)   

train_data = pd.read_csv(f'CV/cross validation/dengue/train.csv')
data_xcoords_train = np.array(train_data['Longitude'], dtype=np.float64) 
data_ycoords_train = np.array(train_data['Latitude'], dtype=np.float64)  
data_coords_train = list(zip(train_data['Longitude'], train_data['Latitude']))    
data_coords_train = np.array(data_coords_train, dtype=np.float64)  

valid_data = pd.read_csv(f'CV/cross validation/dengue/valid.csv')
data_xcoords_valid = np.array(valid_data['Longitude'], dtype=np.float64)   
data_ycoords_valid = np.array(valid_data['Latitude'], dtype=np.float64)   
data_coords_valid = list(zip(valid_data['Longitude'], valid_data['Latitude']))     
data_coords_valid = np.array(data_coords_valid, dtype=np.float64) 

x_temp = []
for name in name_x:
    x_temp.append(np.array(train_data[name], dtype=np.float64).reshape(-1, 1))
x_train = np.hstack(x_temp)   
y_train = np.hstack(train_data[name_y], dtype=np.float64).reshape(-1, 1)

x_temp = []
for name in name_x:
    x_temp.append(np.array(valid_data[name], dtype=np.float64).reshape(-1, 1))
x_valid = np.hstack(x_temp) #  
y_valid = np.hstack(valid_data[name_y], dtype=np.float64).reshape(-1, 1)


feature_num = len(name_x)  
weight_num = feature_num + 1  

# get_nn_dis 
def get_nn_dis(sample_coords, ref_coords_x, ref_coords_y):
    dis_list = [] # 
    for i in range(sample_coords.shape[0]):  
        sample_coords_x = sample_coords[i][0]  
        sample_coords_y = sample_coords[i][1]  
        dis = np.sqrt(np.square(ref_coords_x - sample_coords_x) + np.square(ref_coords_y - sample_coords_y))  
        dis_list.append(dis)  
    return np.array(dis_list)  
 
def my_round(x):
    condition = tf.greater(x, 0.5)  
    return tf.where(condition, 1.0, 0.0)  

# common part ----- distance
def dis_in_gnnwlr(input_dis):
    dis = input_dis  

    dis = tf.keras.layers.Dense(1000, activation='relu', kernel_initializer=tf.initializers.HeNormal(),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(dis)
    dis = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.initializers.HeNormal(),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(dis)
    dis = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=tf.initializers.HeNormal(),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(dis)
    dis = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.initializers.HeNormal(),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(dis)

    # plus 1 is constant 
    dis = tf.keras.layers.Dense(weight_num)(dis)
    return dis  

# x_train_demo, dis_train_demo
x_temp = []
for name in name_x:
    x_temp.append(np.array(train_data[name], dtype=np.float64).reshape(-1, 1))
x_train_demo = np.hstack(x_temp)   
dis_train_demo = get_nn_dis(data_coords_train, data_coords_x, data_coords_y)  

# gnnwlr
def gnnwlr(beta_ols):
    input_dis = tf.keras.layers.Input(shape=dis_train_demo.shape[1:])  
    inputs_x = tf.keras.layers.Input(shape=x_train_demo.shape[1:])  

    dis = dis_in_gnnwlr(input_dis) 

    weights = []  
    for i in range(weight_num):  
        weights.append(dis[:, i])  

    y_pred = weights[0] * beta_ols[0][0]  
    for i in range(1, weight_num):  
        y_pred += weights[i] * beta_ols[i][0] * inputs_x[:, i-1] 

    y_bin = tf.keras.layers.Lambda(lambda z: tf.nn.sigmoid(z))(y_pred)  
    return tf.keras.Model(inputs=(input_dis, inputs_x), outputs=y_bin)  

# this is Spatial Weighted Neural Network Calculate the weights of input locations.
def swnn():
    input_dis = tf.keras.layers.Input(shape=dis_train_demo.shape[1:])  
    inputs_x = tf.keras.layers.Input(shape=x_train_demo.shape[1:]) 

    dis = dis_in_gnnwlr(input_dis)  

    return tf.keras.Model(inputs=(input_dis, inputs_x), outputs=dis)

import os
import datetime

TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")

nn_logdir = './mpm_nn_callbacks'  
if not os.path.exists(nn_logdir):   
    os.mkdir(nn_logdir)
nn_output_model_path = os.path.join(nn_logdir, "gnnwlr_model_{}.keras".format(TIMESTAMP))  

# nn_optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-07,name='Adagrad', )
# nn_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005) # 0.96 0.71 0.86 0.66
# nn_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.708, nesterov=True)
nn_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001) # 0.95 0.87 0.85 0.78

def nn_scheduler(epoch, lr): 
    if epoch < 80: 
        return 0.001
    else: 
        return lr * tf.math.exp(-0.005)

 
nn_callbacks = [
    tf.keras.callbacks.TensorBoard(nn_logdir), 
    tf.keras.callbacks.ModelCheckpoint(nn_output_model_path, monitor='val_loss', save_best_only=True, mode='auto'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.010, 
                                     patience=15, restore_best_weights=True, mode='auto') # 
]


#### cross validation
from pysal.model.spreg import OLS
import datetime
from sklearn import metrics
 
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

EPOCHS = 500 
NN_BATCH_SIZE = 32  
NN_BUFFER_SIZE = 4096 

AUC_train_result = []
AUC_valid_result = []
ACC_train_result = []
ACC_valid_result = []

print(f"====Cross-Validation =====") 
# clear the session
tf.keras.backend.clear_session() 
# prepare data 
dis_train = get_nn_dis(data_coords_train, data_coords_x, data_coords_y)
dis_valid = get_nn_dis(data_coords_valid, data_coords_x, data_coords_y)

nn_train_input = tf.data.Dataset.from_tensor_slices((dis_train, x_train))
nn_train_output = tf.data.Dataset.from_tensor_slices(y_train)
nn_train = tf.data.Dataset.zip((nn_train_input, nn_train_output))
nn_train = nn_train.shuffle(NN_BUFFER_SIZE).batch(NN_BATCH_SIZE)

nn_valid_input = tf.data.Dataset.from_tensor_slices((dis_valid, x_valid))
nn_valid_output = tf.data.Dataset.from_tensor_slices(y_valid)
nn_valid = tf.data.Dataset.zip((nn_valid_input, nn_valid_output))
nn_valid = nn_valid.batch(NN_BATCH_SIZE)

# construct model
ols = OLS(y_train, x_train, name_y=name_y, name_x=name_x)
beta_ols = ols.betas
print(f'beta_0 in ols is {beta_ols[0][0]}')

# calling the model and compile it 
gnnwlr_model = gnnwlr(beta_ols)
gnnwlr_model.compile(optimizer=nn_optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

# run the model
nn_history = gnnwlr_model.fit(nn_train, validation_data=nn_valid, epochs=EPOCHS, callbacks=nn_callbacks)

nn_y_train = gnnwlr_model.predict((dis_train, x_train)).reshape(-1, 1)
nn_y_train_0_1 = np.where(nn_y_train > 0.5, 1, 0)
nn_y_valid = gnnwlr_model.predict((dis_valid, x_valid)).reshape(-1, 1)
nn_y_valid_0_1 = np.where(nn_y_valid > 0.5, 1, 0)
    
#  ACC AUC for dataset
train_auc = metrics.roc_auc_score(y_train, nn_y_train)
valid_auc = metrics.roc_auc_score(y_valid, nn_y_valid)
train_acc = metrics.accuracy_score(y_train, nn_y_train_0_1)
valid_acc = metrics.accuracy_score(y_valid, nn_y_valid_0_1)
print('y train AUC:', train_auc)
AUC_train_result.append(train_auc)
print('y valid AUC:', valid_auc)
AUC_valid_result.append(valid_auc)
print('y train ACC:', train_acc)
ACC_train_result.append(train_acc)
print('y valid ACC:', valid_acc)
ACC_valid_result.append(valid_acc)

TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")
pd.DataFrame(beta_ols).to_csv(f'./CV/weight/beta_ols_cv.csv', index=False) 

gnnwlr_model.save_weights(f'./CV/weight/gnnwlr_wg_cv.weights.h5') 
print(f'Cross Validation finished.') 


#### GNNWLR estimation for all dataset (ROC、AUC、ACC、Recall) ####
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

beta_ols = np.array(pd.read_csv(f'./CV/weight/beta_ols_cv.csv'))
gnnwlr_model = gnnwlr(beta_ols)
gnnwlr_model.load_weights(f'./CV/weight/gnnwlr_wg_cv.weights.h5')

dis_data = get_nn_dis(data_coords, data_coords_x, data_coords_y)

x_temp = []
for name in name_x:
    x_temp.append(np.array(data[name], dtype=np.float64).reshape(-1, 1))
x_all = np.hstack(x_temp) 
y_all = np.hstack(data[name_y], dtype=np.float64).reshape(-1, 1)

nn_pred = gnnwlr_model.predict((dis_data, x_all)).reshape(-1, 1)

all_auc = roc_auc_score(y_all, nn_pred)
print('y all AUC:', all_auc)

acc_all = accuracy_score(y_all, np.where(nn_pred > 0.5, 1, 0))
recall_all = recall_score(y_all, np.where(nn_pred > 0.5, 1, 0))
print(f'all region acc: {acc_all}')
print(f'all region recall: {recall_all}')

data['GNNWLR_pred'] = nn_pred

#### GNNWLR estimation for overall region (China) ####
# distance 
alldata = pd.read_csv('./allregion/dengue/all100585.csv') 
alldata_coords = list(zip(alldata['Longitude'], alldata['Latitude']))
alldata_coords = np.array(alldata_coords, dtype=np.float64)
alldata_coords_x = np.array(alldata['Longitude'], dtype=np.float64)   
alldata_coords_y = np.array(alldata['Latitude'], dtype=np.float64)
dis_alldata = get_nn_dis(alldata_coords, data_coords_x, data_coords_y) 

x_temp = []
for name in name_x:
    x_temp.append(np.array(alldata[name], dtype=np.float64).reshape(-1, 1))
x_all_2 = np.hstack(x_temp)  


# alldata['GNNWLR_pred'] 
import os
from sklearn.metrics import confusion_matrix

beta_ols = np.array(pd.read_csv(f'./CV/weight/beta_ols_cv.csv'))
gnnwlr_model = gnnwlr(beta_ols)
gnnwlr_model.load_weights(f'./CV/weight/gnnwlr_wg_cv.weights.h5')

nn_allpred = gnnwlr_model.predict((dis_alldata, x_all_2)).reshape(-1, 1)

prevalue = pd.DataFrame(nn_allpred)
alldata['GNNWLR_pred'] = prevalue
prevalue
df = pd.DataFrame(prevalue)

df.to_csv('./result/estimation.csv', index=False)

import pandas as pd
for i, name in enumerate(name_x): 
    temp = x_all_2.copy() 
    temp[:, i] = 0 
    temp_pred = gnnwlr_model.predict((dis_alldata, temp)).reshape(-1, 1) 
    alldata[f"lack_{name}"] = temp_pred.squeeze().tolist() 
    alldata[f'SHAP_{name}'] = alldata['GNNWLR_pred'] - alldata[f"lack_{name}"]  

alldata
df = pd.DataFrame(alldata)

# GNNWLR-SHAP estimation
df.to_csv('./result/SHAP.csv', index=False) 