#!/usr/bin/python
## Author: sumanth
## Date: March 10,2017
# test script to fetch the hog features

import hog_features
import pipeLine
import glob
import pickle
import cv2
import matplotlib.pyplot as plt

# test the pipeline on images
images = glob.glob('test_images/test*.jpg')

# loda the model
with open('model.p', 'rb') as f:
    model = pickle.load(f)

svc = model['svc']
X_scaler = model['X_scaler']

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1
    dst = pipeLine.processImage(img)
    plt.clf()
    plt.imshow(dst)
    cv2.imshow('img',dst)
    plt.savefig('output_images/test' + str(count) + '.png')
