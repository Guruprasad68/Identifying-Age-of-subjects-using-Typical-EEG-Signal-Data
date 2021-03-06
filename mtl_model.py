# -*- coding: utf-8 -*-
"""eeg_mtl_pkl_200.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ypvg9l-uZFwyjpTPOHY6LLjNMrmutkt0
"""

from google.colab import drive
drive.mount('/content/drive')

#Libraries
import pickle
import pandas as pd
import glob
import numpy as np
import re
import os
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Input,Conv2D,UpSampling2D,MaxPooling2D,Activation,Flatten,Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import  ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import random
from sklearn.metrics import accuracy_score, f1_score,classification_report,confusion_matrix

#Obtaining the train,valid,test data
L={}
with(open('/content/drive/MyDrive/539: Intro to ANN/EEG_Data/Trfolder_200_data.pkl','rb')) as f:
  L=pickle.load(f)
print(L['x_train'].shape,L['x_test'].shape,L['x_val'].shape)
x_train_1=np.reshape(L['x_train'],(2880,32,40))
x_valid_1=np.reshape(L['x_val'],(720,32,40))
x_test_1=np.reshape(L['x_test'],(1000,32,40))
y_train_1=np.array(L['y_train'])
y_valid_1=np.array(L['y_val'])
y_test_1=np.array(L['y_test'])
print(y_train_1.shape,y_test_1.shape,y_valid_1.shape)
x_train_2=np.moveaxis(x_train_1,1,2).reshape((2880,40,32,1))
x_valid_2=np.moveaxis(x_valid_1,1,2).reshape((720,40,32,1))
x_test_2=np.moveaxis(x_test_1,1,2).reshape((1000,40,32,1))

print(x_train_2.shape,x_test_2.shape,x_valid_2.shape)
y_train_2_one_hot=tf.keras.utils.to_categorical(y_train_1,num_classes=5)
y_valid_2_one_hot=tf.keras.utils.to_categorical(y_valid_1,num_classes=5)
y_test_2_one_hot=tf.keras.utils.to_categorical(y_test_1,num_classes=5)

X_train=x_train_2; X_valid=x_valid_2; X_test=x_test_2
Y_train=y_train_2_one_hot;Y_valid=y_valid_2_one_hot; Y_test=y_test_2_one_hot

print(X_train.shape,Y_train.shape)
print(X_valid.shape,Y_valid.shape)
print(X_test.shape,Y_test.shape)

np.random.seed(12345)
tf.random.set_seed(12345)
random.seed(12345)

#Preprocessing the unseen data
UNS={}
with(open('/content/drive/MyDrive/539: Intro to ANN/EEG_Data/Evalastestset.pkl','rb')) as f:
  UNS=pickle.load(f)
x_unseen_set1_1=UNS['X_test1']
x_unseen_set2_1=UNS['X_test2']
x_unseen_set3_1=UNS['X_test3']
x_unseen_set4_1=UNS['X_test4']
print(x_unseen_set1_1.shape,x_unseen_set2_1.shape,x_unseen_set3_1.shape,x_unseen_set4_1.shape)
x_unseen_set1_2=np.reshape(x_unseen_set1_1,(552,32,40)) 
x_unseen_set2_2=np.reshape(x_unseen_set2_1,(552,32,40))
x_unseen_set3_2=np.reshape(x_unseen_set3_1,(552,32,40))
x_unseen_set4_2=np.reshape(x_unseen_set4_1,(460,32,40))
X_unseen1=np.moveaxis(x_unseen_set1_2,1,2).reshape((552,40,32,1))
X_unseen2=np.moveaxis(x_unseen_set2_2,1,2).reshape((552,40,32,1))
X_unseen3=np.moveaxis(x_unseen_set3_2,1,2).reshape((552,40,32,1))
X_unseen4=np.moveaxis(x_unseen_set4_2,1,2).reshape((460,40,32,1))

Y_unseen1=tf.keras.utils.to_categorical(np.array(UNS['Y_test1']),num_classes=5)
Y_unseen2=tf.keras.utils.to_categorical(np.array(UNS['Y_test2']),num_classes=5)
Y_unseen3=tf.keras.utils.to_categorical(np.array(UNS['Y_test3']),num_classes=5)
Y_unseen4=tf.keras.utils.to_categorical(np.array(UNS['Y_test4']),num_classes=5)

print(X_unseen1.shape,X_unseen2.shape,X_unseen3.shape,X_unseen4.shape)
print(Y_unseen1.shape,Y_unseen2.shape,Y_unseen3.shape,Y_unseen4.shape)

X_unseen=np.concatenate((X_unseen1,X_unseen2,X_unseen3,X_unseen4))
Y_unseen=np.concatenate((Y_unseen1,Y_unseen2,Y_unseen3,Y_unseen4))
print(X_unseen.shape,Y_unseen.shape)


#MTL CNN Autoencoder/Age Classifier:Iteration 2, 2 Layer Encoder and Decoder without regularization and loss weights(0.8,0.2)

#Encoder portion
input=Input(shape=X_train.shape[1:])

enc1=Conv2D(8,(3,3),padding='same')(input)
enc1=BatchNormalization()(enc1)
enc1=Activation('relu')(enc1)
enc1=MaxPooling2D((2,2),padding='same')(enc1)

enc2=Conv2D(16,(3,3),padding='same')(enc1)
enc2=BatchNormalization()(enc2)
enc2=Activation('relu')(enc2)
enc2_bn=MaxPooling2D((2,2))(enc2)


#Decoder Portion

dec1=Conv2D(16,(3,3),padding='same')(enc2_bn)
dec1=BatchNormalization()(dec1)
dec1=Activation('relu')(dec1)
dec1=UpSampling2D((2,2))(dec1)

dec2=Conv2D(8,(3,3),padding='same')(dec1)
dec2=BatchNormalization()(dec2)
dec2=Activation('relu')(dec2)
dec2=UpSampling2D((2,2))(dec2)
dec2_out=Conv2D(1,(1,1),padding='same',name='reconstruction')(dec2)

#Classifier Portion
cl_l1=Flatten()(enc2_bn)
cl_l2=Dense(100,activation='relu')(cl_l1)
cl_l3=Dense(100,activation='relu')(cl_l2)
cl_l4=Dense(5,activation='softmax',name='age_group')(cl_l3)

#Defining Model
model_MTL_cnnae_cl_iter2=keras.Model(input,outputs=[dec2_out,cl_l4])
model_MTL_cnnae_cl_iter2.summary()

tf.keras.utils.plot_model(model_MTL_cnnae_cl_iter2,show_shapes=True)

model_MTL_cnnae_cl_iter2.compile(optimizer='adam',loss={'age_group':'categorical_crossentropy','reconstruction':'mse'},loss_weights={'reconstruction':0.8,'age_group':0.2},
                        metrics=['accuracy'])
checkpoint_MTL_cnnae_cl_iter2=ModelCheckpoint('/content/drive/MyDrive/MTL_CNNAE_classifier_pkl_200_iter2.ckpt',verbose=1,save_best_only=True,monitor='val_age_group_accuracy',mode='max')
history_MTL_cnnae_cl_iter2=model_MTL_cnnae_cl_iter2.fit(X_train,{'age_group':Y_train,'reconstruction':X_train},validation_data=[X_valid,{'age_group':Y_valid,'reconstruction':X_valid}],epochs=200,callbacks=[checkpoint_MTL_cnnae_cl_iter2])

model_MTL_cnnae_cl_iter2.evaluate(X_test,{'age_group':Y_test,'reconstruction':X_test})
model_MTL_cnnae_cl_iter2_test=keras.models.load_model('/content/drive/MyDrive/MTL_CNNAE_classifier_pkl_200_iter2.ckpt')
model_MTL_cnnae_cl_iter2_test.evaluate(X_test,{'age_group':Y_test,'reconstruction':X_test})



print(classification_report(np.argmax(Y_test,axis=1),np.argmax(model_MTL_cnnae_cl_iter2_test.predict(X_test)[1],axis=1)))

plt.plot(history_MTL_cnnae_cl_iter2.history['loss'])
plt.plot(history_MTL_cnnae_cl_iter2.history['val_loss'])
plt.title('Loss Curve')
plt.show()

plt.plot(history_MTL_cnnae_cl_iter2.history['age_group_accuracy'])
plt.plot(history_MTL_cnnae_cl_iter2.history['val_age_group_accuracy'])
plt.title('Accuracy Curve')
plt.show()
print(confusion_matrix(np.argmax(Y_test,axis=1),np.argmax(model_MTL_cnnae_cl_iter2_test.predict(X_test)[1],axis=1)))

#Testing on Unseen data
model_MTL_cnnae_cl_iter2_test.evaluate(X_unseen1,{'age_group':Y_unseen1,'reconstruction':X_unseen1})
model_MTL_cnnae_cl_iter2_test.evaluate(X_unseen2,{'age_group':Y_unseen2,'reconstruction':X_unseen2})
model_MTL_cnnae_cl_iter2_test.evaluate(X_unseen3,{'age_group':Y_unseen3,'reconstruction':X_unseen3})
model_MTL_cnnae_cl_iter2_test.evaluate(X_unseen4,{'age_group':Y_unseen4,'reconstruction':X_unseen4})
model_MTL_cnnae_cl_iter2_test.evaluate(X_unseen,{'age_group':Y_unseen,'reconstruction':X_unseen})
