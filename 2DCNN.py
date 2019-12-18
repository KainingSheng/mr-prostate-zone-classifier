# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:58:20 2019

@author: KainingSheng

2D CNN for MRI zone prostate detection

This work is a 2D conversion of the model architecture found in: 
    Aldoj, N., Lukas, S., Dewey, M., Penzkofer, T., 2019. 
    Semi-automatic classification of prostate cancer on multi-parametric MR imaging using a multi-channel 3D convolutional neural network. 
    Eur. Radiol. https://doi.org/10.1007/s00330-019-06417-z

"""
import h5py
import keras
import tensorflow as tf

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Concatenate, concatenate
from keras.layers import Dropout, Input, BatchNormalization
from keras.activations import relu
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2


def getModel():
    ## input layer
    input_layer = Input((96, 96,1))
    
    ## Block 1
    conv_layer1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_layer1 = BatchNormalization()(conv_layer1)
    pooling_layer1 = MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')(batch_layer1)
    
    
    ## Block 2
    conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling_layer1)
    batch_layer2 = BatchNormalization()(conv_layer2)
    
    conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_layer2)
    batch_layer3 = BatchNormalization()(conv_layer3)
    
    concat_2 = Concatenate(axis=-1)([pooling_layer1, batch_layer3])
    pooling_layer2 = MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')(concat_2)
    
    
    ## Block 3
    conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pooling_layer2)
    batch_layer4 = BatchNormalization()(conv_layer4)
    
    conv_layer5 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(batch_layer4)
    batch_layer5 = BatchNormalization()(conv_layer5)
    
    concat_3 = Concatenate(axis=-1)([pooling_layer2, batch_layer5])
    pooling_layer3 = MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')(concat_3)
    
    
    ## Block 4
    conv_layer6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling_layer3)
    batch_layer6 = BatchNormalization()(conv_layer6)
    
    conv_layer7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_layer6)
    batch_layer7 = BatchNormalization()(conv_layer7)
    
    concat_4 = Concatenate(axis=-1)([pooling_layer3, batch_layer7])
    pooling_layer4 = MaxPool2D(pool_size=(2, 2), padding='same')(batch_layer7)
    
    
    ## Block 5
    conv_layer8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling_layer4)
    batch_layer8 = BatchNormalization()(conv_layer8)
    
    conv_layer9 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_layer8)
    batch_layer9 = BatchNormalization()(conv_layer9)
     
    concat_5 = Concatenate(axis=-1)([pooling_layer4, batch_layer9])
    pooling_layer5 = MaxPool2D(pool_size=(2, 2), padding='same')(concat_5)
    
    
    ## Block 6
    conv_layer10 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling_layer5)
    batch_layer10 = BatchNormalization()(conv_layer10)
    
    conv_layer11 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_layer10)
    batch_layer11 = BatchNormalization()(conv_layer11)
    
    concat_6 = Concatenate(axis=-1)([pooling_layer5, batch_layer11])
    pooling_layer6 = MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')(concat_6)
    
    
    ## Block 7
    conv_layer12 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pooling_layer6)
    batch_layer12 = BatchNormalization()(conv_layer12)
    
    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    pooling_layer12 = MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same')(batch_layer12)
    flatten_layer = Flatten()(pooling_layer12)
    
    
    ## create an MLP architecture with dense layers : 2048 -> 512 -> 2
    ## add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=2048, activation='relu',kernel_regularizer=l2(1e-4),kernel_initializer='random_uniform')(flatten_layer)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    dense_layer2 = Dense(units=512, activation='relu',kernel_regularizer=l2(1e-4),kernel_initializer='random_uniform')(dense_layer1)
    dense_layer2 = Dropout(0.5)(dense_layer2)
    #output_layer = Dense(units=1, activation='sigmoid')(dense_layer2)
    output_layer = Dense(units=2, activation='softmax')(dense_layer2)
    
    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
    