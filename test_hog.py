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

###############################################################################
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
    plt.savefig('output_images/test' + str(count) + '.jpg')

################################################################################
# hog features on test images
# load the data

cars = glob.glob('vehicles/*/*/*.png')
nocars = glob.glob('non-vehicles/*/*/*.png')

img1 = cv2.imread(cars[10])
img2 = cv2.imread(nocars[10])

# hog params
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32 # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
ystart = 400 # Min and max in y to search find_cars()
ystop = 656
scale = 1.5

# car image
plt.clf()
plt.imshow(img1)
plt.savefig('output_images/hog/input1' + '.jpg')

ft, him = hog_features.fetchHOGFeatures(img1[:,:, 0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.clf()
plt.imshow(him)
plt.savefig('output_images/hog/output11' + '.jpg')

ft, him = hog_features.fetchHOGFeatures(img1[:,:, 1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.clf()
plt.imshow(him)
plt.savefig('output_images/hog/output12' + '.jpg')

ft, him = hog_features.fetchHOGFeatures(img1[:,:, 2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.clf()
plt.imshow(him)
plt.savefig('output_images/hog/output13' + '.jpg')

# no car image
plt.clf()
plt.imshow(img2)
plt.savefig('output_images/hog/input2' + '.jpg')

ft, him = hog_features.fetchHOGFeatures(img2[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.clf()
plt.imshow(him)
plt.savefig('output_images/hog/output21' + '.jpg')

ft, him = hog_features.fetchHOGFeatures(img2[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.clf()
plt.imshow(him)
plt.savefig('output_images/hog/output22' + '.jpg')

ft, him = hog_features.fetchHOGFeatures(img2[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
plt.clf()
plt.imshow(him)
plt.savefig('output_images/hog/output23' + '.jpg')

###############################################################################
# sliding window test

count = 0;
for fname in images:
    img = cv2.imread(fname)
    count +=1
    out_img, detected = hog_features.detectCar(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size,hist_bins)
    print(detected)
    plt.clf()
    plt.imshow(out_img)
    plt.savefig('output_images/sliding/op' + str(count) + '.jpg')
