#!/usr/bin/python
## Author: sumanth
## Date: March 10,2017
# main file running the vehicle detcetion and tracking pipeline


import pickle
from scipy.ndimage.measurements import label
import classifier
import hog_features
import numpy as np
from moviepy.editor import VideoFileClip

# parameters
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

# train the classifier
#classifier.trainClassifier(color_space, spatial_size, hist_bins, orient,
#1                             pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

#classifier.trainClassifier()
# loda the model
with open('model.p', 'rb') as f:
    model = pickle.load(f)

svc = model['svc']
X_scaler = model['X_scaler']

# process each image
def processImage(img):
    image_copy = np.copy(img)
    out_img, detected = hog_features.detectCar(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size,hist_bins)


    heatmap = np.zeros(out_img.shape)
    heatmap = hog_features.addHeatMap(heatmap, detected)
    heatmap = hog_features.applyThreshold(heatmap, 0)
    labels = hog_features.label(heatmap)

    # Draws bounding boxes on a copy of the image
    draw_img = hog_features.drawLabeledBoundingBoxes(np.copy(image_copy), labels)

    return draw_img
