import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session  
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  
set_session(tf.compat.v1.Session(config=config))

import sklearn
from sklearn.metrics import classification_report
import numpy as np, matplotlib.pyplot as plt, pandas as pd, pickle, librosa as lr, os
import librosa.display as lrd

from tensorflow.keras import regularizers
import tensorflow as tf
import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, CSVLogger 
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.losses import kullback_leibler_divergence
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.utils import plot_model
def get_lr_metric(optimizer):
  def lr(y_true, y_pred):
      return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
  return lr

print('Imported packages and modules successfully!')

path = '___/'
files_path = path + 'Data/'
# L = os.listdir(files_path)
if os.path.isfile(files_path + 'Trdata32_CNN1D_200.pkl') == True:
  with open(files_path + 'Trdata32_CNN1D_200.pkl', 'rb') as f:
    L = pickle.load(f)
  X_train = L['x_train'].squeeze(); X_val = L['x_val'].squeeze(); X_test = L['x_test'].squeeze()
  Y_train = L['y_train']; Y_val = L['y_val']; Y_test = L['y_test']
    
B_I = False
num_classes = 5 #[0<=age<10, 11<=age<21, 21<=age<=31, 31<=age<41, 41<=age<51, 51<=age<61, 61<=age<71, 71<=age<81, 81<=age<91, 91<=age<101]
Y_train = to_categorical(Y_train, num_classes = num_classes)
Y_val = to_categorical(Y_val, num_classes = num_classes)
Y_test = to_categorical(Y_test, num_classes = num_classes)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation = "relu"))
model.add(Dropout(0.4))
model.add(BatchNormalization())

#model.add(Dense(512, activation = "relu"))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation = "relu"))
#model.add(Dropout(0.2))
model.add(BatchNormalization())

#model.add(Dense(64, activation = "relu"))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(5, activation = "softmax"))
print("[INFO] training network...")
learning_rate = 0.01
num_epochs = 200

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=100,
    decay_rate=0.95)

opt = optimizers.Adam(learning_rate = lr_schedule)
#opt = optimizers.Adam(learning_rate = lr_schedule, epsilon=None, decay=1e-6)
lr_metric = get_lr_metric(opt)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy", lr_metric ])
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
	epochs=num_epochs, batch_size=1024)

print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=1024)
print(classification_report(Y_test.argmax(axis=1),
	predictions.argmax(axis=1)))
print(H.history.keys())
print(np.mean(H.history['val_accuracy']), np.mean(H.history['accuracy']))
print(precision_recall_fscore_support(Y_test.argmax(axis=1),predictions.argmax(axis=1)))

model.save(files_path+'atm7sv2.h5')
p,r,f,s = precision_recall_fscore_support(Y_test.argmax(axis=1),predictions.argmax(axis=1))
resdict = {'history':H.history, 'preds':predictions, 'y_test': Y_test, 'prfs': [p,r,f,s]}
with open(files_path + 'resat7sv2.pkl', 'wb') as f:
  pickle.dump(resdict, f)