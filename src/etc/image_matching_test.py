#for image matching try this 
# https://stackoverflow.com/questions/71764494/find-closest-match-of-image-to-10-000-others-with-similar-features

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract(img):
    img = img.resize((224, 224)) # Resize the image
    img = img.convert('RGB') # Convert the image color space
    x = image.img_to_array(img) # Reformat the image
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)[0] # Extract Features
    return feature / np.linalg.norm(feature)

# Iterate through images and extract Features
images = ["img1.png","img2.png","img3.png","img4.png","img5.png"...+2000 more]
all_features = np.zeros(shape=(len(images),4096))

for i in range(len(images)):
    feature = extract(img=Image.open(images[i]))
    all_features[i] = np.array(feature)

# Match image
query = extract(img=Image.open("image_to_match.png")) # Extract its features
dists = np.linalg.norm(all_features - query, axis=1) # Calculate the similarity (distance) between images
ids = np.argsort(dists)[:5] # Extract 5 images that have lowest distance

