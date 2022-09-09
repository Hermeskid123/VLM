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
    print(num_with_zeros)
    return num_with_zeros

def get_random_images():
    path = "/home/preston/Documents/dataset/train"
    os.chdir(path)
    ran = random_with_0_digits()
    random_num = "0"+str(ran)
    depth_name ="depth_3_0018190.png"
    rgb_name = "rgb_1_"+str(random_num)+".png"
    synth_name = "synthdepth_1_"+str(random_num)+".png"
    return depth_name,rgb_name,synth_name

def create_image_plots():
    depth_name,rgb_name,synth_name = get_random_images()
    imgD = cv.imread(depth_name, 0)
    imgR = cv.imread(rgb_name, 0)
    imgS = cv.imread(synth_name, 0)
 
    stereo = cv.StereoBM_create(numDisparities = 16,blockSize = 15)
 
    disparity = stereo.compute(imgR, imgD)
 
    plt.imshow(imgD)
    plt.subplot(1, 3, 1)
    plt.imshow(imgR, 'gray')
    plt.subplot(1, 3, 2)
    plt.imshow(imgS, 'gray')
    plt.subplot(1, 3, 3)
    plt.show()   


def main():
    create_image_plots()

if __name__ == "__main__":
    sys.exit(main())
