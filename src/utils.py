import math
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
    depth_name ="depth_1_"+str(random_num)+".png"
    rgb_name = "rgb_1_"+str(random_num)+".png"
    synth_name = "synthdepth_1_"+str(random_num)+".png"
    return depth_name,rgb_name,synth_name

