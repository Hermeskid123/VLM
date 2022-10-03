import numpy as npy
image = npy.load("first_img.npy")
print(" number of non-zeros", len(npy.nonzero(image)[0]) )
