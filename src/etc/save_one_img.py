import numpy as np
from PIL import Image
import sys
from matplotlib import pyplot as plt

def load_image( infilename,debug=True ) :
    '''
    returns: numpy array
    '''
    img = Image.open( infilename )
    img.load()
    
    if debug:
        print("Debug: load_image showing img as png")
        img.show()
    
    #data = np.asarray( img, dtype="int32" )
    # you can specify the data type however int 32 breaks the conversion from numpy<->Image
    # dtype is infered when you do not pass any value in
    data = np.asarray( img)
    _ = Image.fromarray(data)
    return data

def load_np_arr(path,debug=False):
    '''
    returns: Image object from PIL 
    '''
    print("loading np arr")
    image = np.load(path)
    print(image.shape)
    
    data = Image.fromarray(image)
    data.save('load_np_arr.png')
    return data

def show_img(img):
    '''
    inputs: numpy img or Image img
    returns: None
    '''
    print(type(img))
    if(type(img) is np.ndarray):
        print("close the window for the program to keep running")
        plt.imshow(img, interpolation='nearest')
        plt.show()
    else:
    #assumes its a Image 
        img.show()
    


def main(path):
    npy_name = 'img_as_numpy.npy'
    print("path is ",path)
    img_as_np = load_image(path)
    
    print("shape ",img_as_np.shape)
    np.save(npy_name, img_as_np)
    show_img(img_as_np)
    
    load_it_back_and_show = load_np_arr(npy_name) 
    show_img(load_it_back_and_show) 

    
    
if __name__ == "__main__":
    if len(sys.argv)<2: 
        print("please provide a path\npython save_one_img.py [path_to_img]\n")
        sys.exit(1)
    else:
        main(sys.argv[1])
