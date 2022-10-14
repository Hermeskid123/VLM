from PIL import Image as im
import numpy as npy

image = npy.load("all_pics-160-120.npy")
#print(" number of non-zeros", len(npy.nonzero(image)[0]) )
print(image.shape)
array = image[0]
print(array.shape)
data = im.fromarray(array)
data.save('dummy_pic.png')
