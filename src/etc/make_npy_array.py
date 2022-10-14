import numpy as np
import os 
from PIL import Image
#this makes a np arrray of all imgs in RGB 
# it saves it in all_pics.npy

folder = '/home/preston/Documents/dataset/train2/RGB/'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
#test = '/home/preston/Documents/dataset/train2/RGB/rgb_1_0033696.png'
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    print(data)
    np.save('rgb_1_0033696.npy', data)
    return data


_shape = (72757,120, 160, 3)
newsize = (160,120)
all_pics = np.zeros(shape=_shape)

count = 0
for filename in os.listdir(folder):
    infilename = (os.path.join(folder,filename))
    img = Image.open( infilename )
    img.load()
    
    img = img.resize(newsize)
    data = np.asarray( img, dtype="int32" )
    all_pics[count] = data
    count = count + 1
    
    img.close()
np.save('all_pics.npy', all_pics)
print(all_pics.shape)

