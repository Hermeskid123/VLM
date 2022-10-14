import scipy.io
import glob
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.layers import *
import keras.backend as K

print(tf.__version__)

def get_lables_for_train():
    data_dir = "/home/preston/Documents/dataset/train/joint_data.mat"
    mat = scipy.io.loadmat(data_dir)
    joint_names = mat["joint_names"]
    joint_uvd = mat["joint_uvd"]
    joint_xyz = mat["joint_xyz"]

    for i in joint_names[0]:
        print(i[0])

    x = []
    
    shape  = (3, 72757, 14, 3)
    new_arr = np.zeros(shape)
    for i in range(0,72757):
        new_arr[0][i][0] = joint_xyz[0][i][32]
        new_arr[0][i][1] = joint_xyz[0][i][30]
        new_arr[0][i][2] = joint_xyz[0][i][31]
        new_arr[0][i][3] = joint_xyz[0][i][24]
        new_arr[0][i][4] = joint_xyz[0][i][25]
        new_arr[0][i][5] = joint_xyz[0][i][27]
        new_arr[0][i][6] = joint_xyz[0][i][0]
        new_arr[0][i][7] = joint_xyz[0][i][3]
        new_arr[0][i][8] = joint_xyz[0][i][6]
        new_arr[0][i][9] = joint_xyz[0][i][9]
        new_arr[0][i][10] = joint_xyz[0][i][12]
        new_arr[0][i][11] = joint_xyz[0][i][15]
        new_arr[0][i][12] = joint_xyz[0][i][18]
        new_arr[0][i][13] = joint_xyz[0][i][21]
        
    for i in new_arr[0]:
        x.append(i.reshape(1,-1))
    return np.array(x).astype(float)
def custom_loss(y_true, y_pred):
    
    

    loss = K.square(tf.subtract(y_true, y_pred))  # (batch_size, 2)
    
    #loss = loss * [0.3, 0.7]          # (batch_size, 2)
    loss = K.sum(loss, axis=1)        # (batch_size,)           
    loss = loss/42
    return loss
def model_run():

    data_dir = "/home/preston/Documents/dataset/train2/RGB"
    
    batch_size = 1
    img_height = 100
    img_width = 100
    num_elements = 72757
    lables = get_lables_for_train()
    #print(lables.shape)
    targets = tf.convert_to_tensor(lables)
    print("targets ",targets.shape)
    
    train_data = np.load('data/all_pics100-100.npy')
    print(train_data.shape) 
    input_shape = train_data[0].shape

    AlexNet = tf.keras.Sequential([
        tf.keras.Input(shape = input_shape),
        ])
    AlexNet.add(Conv2D(filters=96, input_shape=(100,100,3), kernel_size=(11,11), strides=(4,4), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    AlexNet.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    AlexNet.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))

    AlexNet.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    AlexNet.add(Flatten())
    AlexNet.add(Dense(4096, input_shape=(100,100,3,)))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))

    AlexNet.add(Dense(4096))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))

    AlexNet.add(Dense(1000))
    AlexNet.add(BatchNormalization())
    AlexNet.add(Activation('relu'))
    AlexNet.add(Dropout(0.4))
    AlexNet.add(Dense(42))
    
    AlexNet.compile(
        optimizer='adam',
        loss=custom_loss,
        metrics=[tf.keras.metrics.MeanSquaredError()])
   
    AlexNet.fit(train_data,targets,epochs=25, batch_size=1)

    AlexNet.save('test.keras')

    print("End of model run")

def code_test():
    model_run()

code_test()
