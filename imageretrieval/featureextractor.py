import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

files = os.listdir('../images')

def path(files):
    file_path = []
    for id in tqdm(files):
        file_path.append('../images'+ id)
    filepatharray=np.array(file_path)
    dogimagepath = filepatharray.reshape(-1,1)
    return dogimagepath

def model1():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def features(files):
    features = []
    model=model1()
    for i, item in enumerate(files):
        img = cv2.imread('../images/'+item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299,299)) 
        img = img/255.0
        img = img.reshape(1,299, 299, 3)
        out = model.predict(img)
        features.append(out)
        if i%100==0:
            print(i)
            
    features = np.squeeze(features)
    return features


conc = np.concatenate((path(files),features(files)) , axis=1)
df = pd.DataFrame(conc, index= None , columns= None)
df.to_csv('../examples/features1.csv', index= False)