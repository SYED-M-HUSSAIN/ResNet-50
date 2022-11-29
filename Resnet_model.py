import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation,Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, json, math, librosa
import IPython.display as ipd
import librosa.display
import tensorflow as tf
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from google.colab import drive
from statistics import mode
from google.colab import files
import time
import matplotlib.pyplot as plt
import numpy as np

import pathlib
drive.mount('/content/drive')

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

data_dir = pathlib.Path('/content/drive/MyDrive/Deep_Project/Final_Spectograms')

def ResNet_50(data_dir):
  img_height,img_width=180,180
  batch_size=32
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    label_mode='categorical',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  label_mode='categorical',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  resnet_model = Sequential()

  pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                    input_shape=(180,180,3),
                    pooling='avg',classes=2,
                    weights='imagenet')
  for layer in pretrained_model.layers:
          layer.trainable=False

  resnet_model.add(pretrained_model)
  resnet_model.add(Flatten())
  resnet_model.add(Dense(512, activation='relu'))
  resnet_model.add(Dense(2, activation='softmax'))
  resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
  history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)
  
  
  
  
  
  fig1 = plt.gcf()
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  #plt.axis(ymin=0.4,ymax=1.8)
  plt.grid()
  plt.title('Model Accuracy ReNet50')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'], loc = "lower right")
  
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.grid()
  plt.title('Model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc = "upper left")
  plt.show()

