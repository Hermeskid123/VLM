import math
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
from random import randint



def random_with_0_digits():
    num = randint(1, 10**4)
    num_with_zeros = '{:06}'.format(num)
    num_with_zeros = str(num).zfill(6)
    return num_with_zeros

def get_random_images():
    ran = random_with_0_digits()
    random_num2 = randint(1, 3)
    random_num = "0"+str(ran)
    depth_name ="depth_"+str(random_num2)+"_"+str(random_num)+".png"
    rgb_name = "rgb_"+ str(random_num2)+"_"+str(random_num)+".png"
    synth_name = "synthdepth_"+ str(random_num2)+"_"+str(random_num)+".png"
    print(depth_name,rgb_name,synth_name)
    return depth_name,rgb_name,synth_name


def create_image_plots():
    depth_name,rgb_name,synth_name = get_random_images()
    #this picks a random img is the from of name_x_00xxxxx.png
    #todo later add option to puck name_2 or name_3

    #reads the path for the img
    imgD = cv.imread(depth_name, 0)
    imgR = cv.imread(rgb_name, 0)
    imgS = cv.imread(synth_name, 0)
    print("img depth shape" ,imgD.shape)
    print("img RGB shape" ,imgR.shape)
    print("img SynthD. shape" ,imgS.shape)

    #stereo = cv.StereoBM_create(numDisparities = 16,blockSize = 15)

    #disparity = stereo.compute(imgR, imgD)


    #sub plot first always
    plt.subplot(1, 3, 1)
    plt.imshow(imgD,'gray')
    plt.subplot(1, 3, 2)
    plt.imshow(imgR, 'gray')
    plt.subplot(1, 3, 3)
    plt.imshow(imgS, 'gray')
    plt.show()


