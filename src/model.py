
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
print(tf.__version__)



def model_run():

    data_dir = "/home/preston/Documents/dataset/train"
    
    batch_size = 32
    img_height = 180
    img_width = 180
    
    rows, cols = (3, 72757)
    training_images = [[0]*cols]*rows
    
    print(training_images) 
    ''' 
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    '''and None









def code_test():
    model_run()

code_test()
