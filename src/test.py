import math
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
path = "/home/preston/Documents/dataset/train"
os.chdir(path)
#"synthdepth_1_"+str(random_num)+".png"
imgD ="depth_3_0018190.png"
imgD = cv.imread(imgD, 0)

plt.imshow(imgD)
plt.show()

