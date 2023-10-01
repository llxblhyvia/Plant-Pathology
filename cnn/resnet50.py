# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa


from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from keras import backend as K

import logging

import struct
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, Input, BatchNormalization
# %matplotlib inline
import IPython.core.display         
# setup output image format (Chrome works best)
# IPython.core.display.set_matplotlib_formats("svg")

import sklearn

from sklearn.preprocessing import MultiLabelBinarizer,OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# %matplotlib inline
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import random
from tensorflow.keras.models import Sequential, Model

def run():
    logging.basicConfig()
    df = pd.read_csv('/nfsshare/home/dl03/1.0/train.csv')
    df['labels']=df['labels'].apply( lambda string: string.split(' ') )

    mlb = MultiLabelBinarizer()
    hot_labels = mlb.fit_transform(df['labels'])

    df_labels = pd.DataFrame(hot_labels,columns=mlb.classes_,index=df.index)


    datagen = ImageDataGenerator(rescale=1/255.0,
                                rotation_range=5,
                                zoom_range=0.1,
                                shear_range=0.05,
                                horizontal_flip=True,
                                validation_split=0.3)


    train_generator = datagen.flow_from_dataframe(
        df,
        directory='/nfsshare/home/dl03/1.0/train_images_256',
        subset='training',
        x_col='image',
        y_col='labels',
        target_size=(224,224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=444
        )

    valid_generator = datagen.flow_from_dataframe(
        df,
        directory='/nfsshare/home/dl03/1.0/train_images_256',
        subset='validation',
        x_col='image',
        y_col='labels',
        target_size=(224,224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=444
        )


    K.clear_session()
    random.seed(4487); tf.random.set_seed(4487)

    input_shape= (224,224,3)#Using the shape of (224,224)
    #
    base_model = ResNet50(input_shape=input_shape, include_top=False,weights= "/nfsshare/home/dl03/1.0/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

    from tensorflow.keras.layers import MaxPooling2D,GlobalAveragePooling2D,BatchNormalization,Activation
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #fully connected layer
    x = Dense(64, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    # finally, the softmax for the classifier
    predictions = Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input ,outputs = predictions)
    print(model.summary())



    model = tf.keras.Model(inputs=base_model.input ,outputs = predictions)

    f1 = tfa.metrics.F1Score(num_classes=6, average='macro')
    model.compile(optimizer=keras.optimizers.SGD(lr=0.03, decay=1e-4, momentum=0.8, nesterov=True),
                  loss='binary_crossentropy', metrics=['accuracy', f1])

    accearlystop = keras.callbacks.EarlyStopping(
        monitor=f1,     # look at the validation loss tf2.0 accuracy
        min_delta=0.02,       # threshold to consider as no change
        patience=5,             # stop if  epochs with no change
        verbose=1, mode='max', restore_best_weights= True
    )
    lossearlystop = keras.callbacks.EarlyStopping(
        monitor='val_loss',     # look at the validation loss tf2.0 accuracy
        min_delta=0.02,       # threshold to consider as no change
        patience=5,             # stop if  epochs with no change
        verbose=1, mode='min', restore_best_weights= True
    )
    # callbacks_list = [earlystop]
    lrschedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.05, patience=5, verbose=1)
    callbacks_list = [lrschedule]
    # callbacks_list = [accearlystop,lossearlystop]
    #callbacks_list = []

    history = model.fit_generator(
                train_generator,  # data from generator
                 #steps_per_epoch=1,    # should be number of batches per epoch
                epochs=15,
                callbacks=callbacks_list,
                validation_data=valid_generator,
                #validation_steps = 1,
                verbose=True)

    accname = 'f1_score'
    plot_history(history)

def plot_history(history): 
    fig, ax1 = plt.subplots()
    
    ax1.plot(history.history['loss'], 'r', label="training loss ({:.6f})".format(history.history['loss'][-1]))
    ax1.plot(history.history['val_loss'], 'r--', label="validation loss ({:.6f})".format(history.history['val_loss'][-1]))
    ax1.grid(True)
    ax1.set_xlabel('iteration')
    ax1.legend(loc="best", fontsize=9)    
    ax1.set_ylabel('loss', color='r')
    ax1.tick_params('y', colors='r')

    if accname in history.history:
        ax2 = ax1.twinx()

        ax2.plot(history.history[accname], 'b', label="training f1_score ({:.4f})".format(history.history[accname][-1]))
        ax2.plot(history.history['val_'+accname], 'b--', label="validation f1_score ({:.4f})".format(history.history['val_'+accname][-1]))

        ax2.legend(loc="lower right", fontsize=9)
        ax2.set_ylabel('acc', color='b')        
        ax2.tick_params('y', colors='b')

if __name__ == '__main__':
    run()
