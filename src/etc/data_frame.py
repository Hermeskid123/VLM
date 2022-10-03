import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
import tensorflow as tf
#code  that makes a data frame and saves the frame as test.csv
#this is no longer needed for the project

def get_data_frame(path):
    
    directory = path
 
    files = Path(directory).glob('*')
    paths = []
    
    for file in files:
        paths.append(str(file))

    data_dir = "/home/preston/Documents/dataset/train/joint_data.mat"
    mat = scipy.io.loadmat(data_dir)
    joint_names = mat["joint_names"]
    joint_uvd = mat["joint_uvd"]
    joint_xyz = mat["joint_xyz"]

    for i in joint_names[0]:
        print(i[0])

    x = []
    count = 0
    for i in joint_xyz[0]:
        x.append(i.reshape(1,-1))
    
    ndata = np.array(x).astype(float)
    txt_data = get_data_txt()
    #df_arr = pd.DataFrame({'y_col': [[-173.8927895229915 -173.48015846547483 -171.6180030456852 -169.27005570928267 -166.0036297631926 -163.2513945268601 -134.89280905615198 -139.6982491172481 -142.08873663755395 -142.71205785854724 -139.79567607794823 -138.72950563959571 -107.59874836596933 -112.28019789092463 -117.44631886829255 -120.38633618297058 -120.30382410567894 -118.2797203971621 -79.26199408174915 -82.17239248123451 -85.17404422661554 -88.3955839201163 -90.9052286154129 -93.82876245451038 -45.026639462921544 -50.76546806639777 -61.73166338196545 -73.52186762368655 -95.36401598372503 -126.8480552908251 -161.24440082187982 -126.9386680433992 -125.2202302180937 -150.61704035897375 -131.8418958077814 -145.4155639662447]]
#})
    data = pd.DataFrame({'x_col': paths})
    df_arr = pd.DataFrame({'y_col':txt_data}) 
    df_arr.to_csv("test.csv", encoding='utf-8', index=True)
    print(pd.DataFrame.from_dict(df_arr))
    result = pd.merge(
        data,
        df_arr,
        how='left',
        left_index=True,     
        right_index=True 
    )

    print(pd.DataFrame.from_dict(result))
    return result

def make_data_txt():
    data_dir = "/home/preston/Documents/dataset/train/joint_data.mat"
    mat = scipy.io.loadmat(data_dir)
    joint_xyz = mat["joint_xyz"]
    
    f = open("targets.txt", "w")
    for i in joint_xyz[0]:
        for item in i.T:
            for j in item:
                f.write("%s " % j)
            f.write("\n")
def get_data_txt():
    lines = []
    with open('targets.txt') as f:
        for line in f:
            txt = line.rstrip()
            lst = txt.split(' ')
            float_lst = []
            float_lst = list(np.array(lst, dtype = 'float'))
            #tensor = tf.convert_to_tensor(float_lst)
            lines.append(float_lst)
    return lines
#get_data_frame('/home/preston/Documents/dataset/train2/RGB')


