import math
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
from utils import get_random_images

def create_image_plots():
    depth_name,rgb_name,synth_name = get_random_images()
    #this picks a random img is the from of name_1_00xxxxx.png 
    #this fucntion will pefrom a chdir to the path on my computer 
    # you will need to change that path to suit your dir 
    #todo later add option to puck name_2 or name_3 
    
    #reads the path for the img 
    imgD = cv.imread(depth_name, 0)
    imgR = cv.imread(rgb_name, 0)
    imgS = cv.imread(synth_name, 0)
 
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


def main():
    create_image_plots()

if __name__ == "__main__":
    sys.exit(main())
