#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:39:06 2021

@author: nikola
"""

# Importing the libraries

import numpy as np
from skimage import transform
from PIL import Image

from keras import applications
from keras.models import  Model
from keras.layers import Input, Dense, BatchNormalization, MaxPooling2D, Dropout, Flatten


# Model loading function

class ImagePrediction():
    
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    COLORS = 3
    POOLING_DROPOUT_RATE = 0.2
    FC_DROPOUT_RATE = 0.3
    NUMBER_OF_CLASSES = 102
    FC1_NEURONS = 4096 
    FC2_NEURONS = 1024
    FC3_NEURONS = 512
    WEIGHTS_PATH = 'weights.hdf5'

    def __init__(self):
        pass

    # Function to load the model

    def model_load(self):
        
        vgg_model = applications.VGG19(weights = None,
                                       include_top = False,
                                       input_shape = (self.IMG_WIDTH, self.IMG_HEIGHT, self.COLORS))
        
        for layer in vgg_model.layers[:-5]:
            layer.trainable = False
            
        for layer in vgg_model.layers[1:4]:
            layer.trainable = True
            
        inputs = Input(shape = (self.IMG_WIDTH, self.IMG_HEIGHT, self.COLORS), name = 'image_input')
        output_vgg16_conv = vgg_model(inputs)
        
        x = BatchNormalization()(output_vgg16_conv)
        x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)
        x = Dropout(self.POOLING_DROPOUT_RATE)(x)
        
        x = Flatten()(x)
        x = Dense(self.FC1_NEURONS, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.FC_DROPOUT_RATE)(x)
        
        x = Dense(self.FC2_NEURONS, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.FC_DROPOUT_RATE)(x)
        
        x = Dense(self.FC3_NEURONS, activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.FC_DROPOUT_RATE)(x)
        
        x = Dense(self.NUMBER_OF_CLASSES, activation = 'softmax')(x)
        
        model = Model(inputs = inputs, outputs = x)
        
        model.summary()
        
        try:
            model.load_weights(self.WEIGHTS_PATH)
        except FileNotFoundError:
            print('Weights not found, model is loaded empty.')
        
        return model
    
    
    # Function to feed a single image
    
    def feed_image(self, file_path: str, model: Model):
        
        np_image = Image.open(file_path)
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (self.IMG_WIDTH, self.IMG_HEIGHT, self.COLORS))
        np_image = np.expand_dims(np_image, axis = 0)
        
        pred = model.predict(np_image)
        indices = arr.argsort()[-3:][::-1]
    
        return indices










