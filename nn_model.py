#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:43:33 2021

@author: nikola
"""

# This is a model created for the purpose of classifying flower images belonging to 102 distinct flower species.
# It makes use of the pretrained VGG19 model, along with its 'imagenet' weights. :3
# Image set used : https://www.kaggle.com/rednivrug/flower-recognition-he
# For species recognized by this model, please refer to the labels.json file's contents.


import pandas as pd
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

class ImageModel():
    
    # Preprocessing constants
        
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    COLORS = 3
    ROTATION_RANGE = 40
    SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    LABELS_PATH = 'data/train.csv'
    BATCH_SIZE = 32
    SEED = 42
    
    # Neural network layer parameters
        
    POOLING_DROPOUT_RATE = 0.2
    FC_DROPOUT_RATE = 0.3
    NUMBER_OF_CLASSES = 102
    FC1_NEURONS = 4096 
    FC2_NEURONS = 1024
    FC3_NEURONS = 512
    
    # Hyperparameters and others
        
    LEARNING_RATE = 0.1
    DECAY = 0.95
    EPSILON = 1e-08
    NUMBER_OF_EPOCHS = 12
    
    def __init__(self):
        pass
    
    def load_data(self):
    
        traindf = pd.read_csv(self.LABELS_PATH, dtype = str)
        traindf['image_id'] = traindf['image_id'] + ".jpg"
        
        datagen = ImageDataGenerator(rescale = 1./255.,
                                     rotation_range = self.ROTATION,
                                     width_shift_range = self.SHIFT_RANGE, 
                                     height_shift_range = self.SHIFT_RANGE,
                                     shear_range = self.SHEAR_RANGE,
                                     zoom_range = self.ZOOM_RANGE,
                                     horizontal_flip = True,
                                     vertical_flip = True,
                                     fill_mode = 'nearest')
        
        train_generator = datagen.flow_from_dataframe(dataframe = traindf,
                                                      directory = "data/train/",
                                                      x_col = "image_id",
                                                      y_col = "category",
                                                      subset = "training",
                                                      batch_size = self.BATCH_SIZE,
                                                      seed = self.SEED,
                                                      shuffle = True,
                                                      class_mode = "categorical",
                                                      target_size = (self.IMG_WIDTH, self.IMG_HEIGHT))
        
        return train_generator


# Initialzing transfer learning model
    
    def start_model(self):

        vgg_model = applications.VGG19(weights = 'imagenet',
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
        
        return model

    # After performing cross validation, and experimentation, the following
    # hyperparameters, number of epochs and optimizer have been found effective

    def compile_and_fit_model(self, model: Model, train_generator):
        
    
        model.compile(loss = "categorical_crossentropy",
                      optimizer = optimizers.Adadelta(lr = self.LEARNING_RATE,
                                                      rho = self.DECAY,
                                                      epsilon = self.EPSILON,
                                                      decay = 0.0),
                      metrics = ['accuracy'])
        
        STEP_SIZE_TRAIN = train_generator.n // self.BATCH_SIZE
        
        model.fit_generator(generator = train_generator,
                            steps_per_epoch = STEP_SIZE_TRAIN,
                            epochs = self.NUMBER_OF_EPOCHS,
                            verbose = 1)
    
        model.save_weights('weights.hdf5')










