import pandas as pd
import scipy.io
import glob
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from data_frame import get_data_frame
print(tf.__version__)
RGB = 0
D = 1 
SD = 2

def model_run():

    data_dir = "/home/preston/Documents/dataset/train2/RGB"
    
    batch_size = 1
    img_height = 180
    img_width = 180
    
    rows, cols = (3, 72757)
    training_images = [[0]*cols]*rows
     
    num_elements = 72757
    
    data = get_data_frame(data_dir)

    train_datagen = ImageDataGenerator(rescale=1./255)
    
    test_target = np.zeros([3, 3])



    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    train_generator = train_datagen.flow_from_dataframe(
                dataframe = data,
                directory=data_dir,
                x_col = 'x_col',
                y_col = 'y_col',
                has_ext=False,
                target_size=(96, 96),
                batch_size=1,
                class_mode='raw'
                )
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size


        
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(108)
        ])
    #https://vijayabhaskar96.medium.com/tutorial-on-keras-imagedatagenerator-with-flow-from-dataframe-8bd5776e45c1
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['accuracy'])
   
    model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    #validation_data=valid_generator,
                    #validation_steps=STEP_SIZE_VALID,
                    epochs=1)

    print("End of model run")

def code_test():
    model_run()

code_test()
