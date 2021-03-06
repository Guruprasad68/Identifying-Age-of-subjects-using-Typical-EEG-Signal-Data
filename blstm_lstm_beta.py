# -*- coding: utf-8 -*-
"""BLSTM-LSTM_beta.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kcC4e9S5w7lOTBwmZ02PMmRI3GZGFYvk
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
import re
from google.colab import drive
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
drive.mount('/content/drive')

image_dir =Path('/content/drive/MyDrive/BETA')
filepaths=pd.Series(list(image_dir.glob(r'*.csv')),name='Filepath')
filepaths

X=[]
for i in filepaths:
  drive.mount('/content/drive')
  with open(i,'r') as f:
       data = np.genfromtxt(f, dtype='f4', delimiter=',')
  data=data[::1] # data - sampling the every 10th element ( no of rows reduced to 30000 rows)
  df=pd.DataFrame(data)
  y = df.iloc[0:15000,1:26].to_numpy()
  X.append(y)

X_f=np.concatenate((X[0][0:5000,:], X[1][0:5000,:]), axis=0)
for i in range(2,125):
  X_f=np.concatenate((X_f,X[i][0:5000,:]),axis=0)

X_f.shape

X_f1=X_f.reshape(125,5000,25)

df=pd.read_csv('/content/drive/MyDrive/File1.csv')
df['y'][0]
Y=np.full(shape=(126,1),fill_value=0,dtype=np.int16)
for i in range(len(df['y'])):
  if(df['y'][i]>=10 and df['y'][i]<=30):
    Y[i]=0
  elif(df['y'][i]>=31 and df['y'][i]<=40):
    Y[i]=1
  elif(df['y'][i]>=41 and df['y'][i]<=50):
    Y[i]=2
  elif(df['y'][i]>=51 and df['y'][i]<=60):
    Y[i]=3
  elif(df['y'][i]>=61 and df['y'][i]<=90):
    Y[i]=4
print(np.unique(Y,return_counts=True))

Y_reduced_one_hot=tf.keras.utils.to_categorical(Y[0:125],num_classes=5)

X_train,X_test,Y_train,Y_test=train_test_split(X_f1,Y[0:125],test_size=0.2,stratify=Y_reduced_one_hot)

Y_reduced_one_hot_train=tf.keras.utils.to_categorical(Y_train,num_classes=5)
Y_reduced_one_hot_test=tf.keras.utils.to_categorical(Y_test,num_classes=5)

X_train.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils.vis_utils import plot_model

timesteps = 5000
channels_num = 25
model = Sequential()
model.add(Bidirectional(LSTM(256,return_sequences=True),input_shape=(timesteps, channels_num)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(128,activation='tanh',return_sequences=True))
model.add(BatchNormalization())
model.add(LSTM(64,activation='tanh',return_sequences=True))
model.add(BatchNormalization())
model.add(Dense(32,activation='sigmoid'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

checkpoint_filepath = '/content/drive/MyDrive/HW12'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(X_train,Y_reduced_one_hot_train,validation_split=0.2,epochs=50,batch_size=32,callbacks=[model_checkpoint_callback])

model.save_weights('/content/drive/MyDrive/HW12/model_beta.h')

model.evaluate(X_test,Y_reduced_one_hot_test)

model.evaluate(X_train,Y_reduced_one_hot_train)