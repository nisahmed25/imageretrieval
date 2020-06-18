import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import argparse
import matplotlib.pyplot as plt
                                                                                


argparser = argparse.ArgumentParser(description='parse args')
argparser.add_argument('--filepath', type= str, help='add filepath')


class similarimage():
    def __init__(self, args):
        self.filepath = args.filepath
        self.model = self.model1()
    
    def model1(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        self.model = Model(inputs=base_model.input, outputs=x)
        return self.model

    def features(self):
        featurestest = []
        img = cv2.imread(self.filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299,299)) 
        img = img/255.0
        img = img.reshape(1,299, 299, 3)
        out = self.model.predict(img)
        featurestest.append(out)
        features1 = np.squeeze(featurestest)
        features=features1.reshape(1,-1)
        return features


def main(args):
    sim = similarimage(args)
    df = pd.read_csv('../examples/features1.csv')
    df1 =df.drop(labels='0', axis=1)
    embeddings = df1.to_numpy()
    dist = (embeddings - sim.features())**2
    dist = np.sqrt(np.sum(dist, axis=1))
    dfdist = pd.DataFrame({'distance':dist})
    dfdist.sort_values('distance', ascending=True, inplace=True)
    ind=list(dfdist.index)[0]
    img = df.iloc[ind][0]
    return img


if __name__ == '__main__':
    args = argparser.parse_args()
    imgpth=main(args)
    imgpth = str('.')+imgpth
    print('similar image path is : '+ imgpth)
    image= cv2.imread(imgpth)
    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('image',image)
    cv2.waitKey()
