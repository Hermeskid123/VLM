import math
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import sys
from utils import create_image_plots
import scipy.io

def show_mat_data(): 
    mat = scipy.io.loadmat('joint_data.mat')
    #print(mat)
    # I got the joint_name joint_uvd and joint_xyz global vars from the print(mat) out put in the consol

    joint_names = mat["joint_names"]
    joint_uvd = mat["joint_uvd"]
    joint_xyz = mat["joint_xyz"]

    print("joint_name shape ",joint_names.shape)
    print("joint_uvd shape ",joint_uvd.shape)
    print("joint_xyz shape ",joint_xyz.shape)
    print("\nlist of joint names")
    
    #print(joint_names[0]) 
    #use print if you want more than just names
    c = 0
    for i in joint_names[0]:
        print(i,c)
        c += 1
        

    print("\nwhat are these vecotrs for?") 
    print(joint_xyz[0][9422][34][0])
    print(joint_xyz[0][9422][34][1])
    print(joint_xyz[0][9422][34][2])
    #im assuming [Kinects][image_number][joint_name][rgb/depth/synth]


def show_mat_data2(): 
    mat = scipy.io.loadmat('test_predictions.mat')
    print(mat)
    # I got the joint_name joint_uvd and joint_xyz global vars from the print(mat) out put in the consol

    #joint_names = mat["joint_names"]
    joint_names = mat["conv_joint_names"]
    #joint_uvd = mat["joint_uvd"]
    #joint_xyz = mat["joint_xyz"]

    #print("joint_name shape ",joint_names.shape)
    #print("joint_uvd shape ",joint_uvd.shape)
   # print("joint_xyz shape ",joint_xyz.shape)
    print("\nlist of joint names")
    
    #print(joint_names[0]) 
    #use print if you want more than just names
    
    for i in joint_names[0]:
        print(i) 

    print("\nwhat are these vecotrs for?") 
    #print(joint_xyz[0][9422][34][0])
    #print(joint_xyz[0][9422][34][1])
    #print(joint_xyz[0][9422][34][2])
    #im assuming [Kinects][image_number][joint_name][rgb/depth/synth]


def main():
    path = "/home/preston/Documents/dataset/train"
    #change this for your dir 
    os.chdir(path)
    show_mat_data()
    print()
    create_image_plots()

if __name__ == "__main__":
    sys.exit(main())
