#Set the path of files and model
cd D:\Desktop

import pandas as pd
import numpy as np
import tensorflow as tf
import keras

#Read input data file and preprocessing
raw_data = pd.read_csv('Dataset_for_model.csv', encoding = 'cp949')
data_X = raw_data[["O3_sc","NO2_sc","CO_sc","SO2_sc","SZA_sc","T_sc","RH_sc","WS_sc","WD_sc"]]
final_val_X = data_X.dropna(axis=0)    
final_val_X1 = final_val_X.to_numpy().astype('float64')

#Load model
model = keras.models.load_model('USNDv1.0.h5')


final_val_Y1 = model.predict(final_val_X1)
 