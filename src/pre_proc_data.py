import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

rgb = "rgb_1"
sd = "synthdepth_1"
d = "depth_1"

path_rgb = "/home/preston/Documents/dataset/train2/RGB"
path_d = "/home/preston/Documents/dataset/train2/D"
path_sd ="/home/preston/Documents/dataset/train2/SD"

path_rgb2 = "/home/preston/Documents/dataset/test2/RGB"
path_d2 = "/home/preston/Documents/dataset/test2/D"
path_sd2 ="/home/preston/Documents/dataset/test2/SD"


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if(filename[0:len(rgb)] == rgb): 
            #shutil.copy(folder+"/"+filename, path_rgb)
            print(filename[0:len(rgb)])
        if(filename[0:len(sd)] == sd): 
            shutil.copy(folder+"/"+filename, path_sd)
        if(filename[0:len(d)] == d): 
            shutil.copy(folder+"/"+filename, path_d)


    return images

def load_images_from_folder2(folder):
    images = []
    for filename in os.listdir(folder):
        if(filename[0:len(rgb)] == rgb): 
            shutil.copy(folder+"/"+filename, path_rgb2)
            print(filename[0:len(rgb)])
        if(filename[0:len(sd)] == sd): 
            shutil.copy(folder+"/"+filename, path_sd2)
        if(filename[0:len(d)] == d): 
            shutil.copy(folder+"/"+filename, path_d2)


    return images





def code_test():
    data_dir = "/home/preston/Documents/dataset/train"
    load_images_from_folder(data_dir)
    data_dir = "/home/preston/Documents/dataset/test"
    load_images_from_folder2(data_dir)

code_test()
