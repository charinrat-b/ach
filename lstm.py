import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.callbacks import EarlyStopping

from keras.preprocessing import timeseries_dataset_from_array

input_data = df_final[:-7000]
targets = df_final[7000:]
dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    input_data, targets, sequence_length=4)

for x in tqdm(dataset):
    training, label= x
    
# splitting dataset
X_train, X_test, y_train, y_test = X[:84], X[84:], y[:84], y[84:]

from keras import Sequential
from keras.layers import Dense, LSTM

model_uni = Sequential()
model_uni.add(LSTM(200, return_sequences= True, activation='relu', input_shape=(4,7)))
model_uni.add(LSTM(150))
model_uni.add(Dense(7))

print(model_uni.summary())
model_uni.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model_unit.h5', verbose=1, save_best_only=True)
]



history_uni = model_uni.fit(X_train, y_train, epochs=100,validation_data=(X_test,y_test), batch_size=32, callbacks=callbacks)


acc = history_uni.history['accuracy']
val_acc = history_uni.history['val_accuracy']
loss = history_uni.history['loss']
val_loss = history_uni.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
