#!/usr/bin/python
## Author: sumanth
## Date: March 10,2017
# main file running the vehicle detcetion and tracking pipeline

import numpy as np
import cv2
import glob
import time
import hog_features
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label


# read the data
cars = glob.glob('vehicles/*/*/*.png')
nocars = glob.glob('non-vehicles/*/*/*.png')
print(len(cars))
print(len(nocars))

# extract hog features
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 5  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 5 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 24    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

# split size
split_size=0.2 # 20%

car_features = hog_features.extractFeatures(cars, color_space=color_space,           \
                        spatial_size=spatial_size, hist_bins=hist_bins,     \
                        orient=orient, pix_per_cell=pix_per_cell,           \
                        cell_per_block=cell_per_block,                      \
                        hog_channel=hog_channel, spatial_feat=spatial_feat, \
                        hist_feat=hist_feat, hog_feat=hog_feat)

nocar_features = hog_features.extractFeatures(nocars, color_space=color_space,      \
                        spatial_size=spatial_size, hist_bins=hist_bins,     \
                        orient=orient, pix_per_cell=pix_per_cell,           \
                        cell_per_block=cell_per_block,                      \
                        hog_channel=hog_channel, spatial_feat=spatial_feat, \
                        hist_feat=hist_feat, hog_feat=hog_feat)

# stack up the features into a vector
X = np.vstack((car_features, nocar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Normalise input
X = normalise(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(nocar_features))))

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, split_size=0.2, random_state=42)

print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
