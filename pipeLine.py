#!/usr/bin/python
## Author: sumanth
## Date: March 10,2017
# main file running the vehicle detcetion and tracking pipeline

import numpy as np
import cv2
import glob
import time
import hog_features
import slidingWindow
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
X = hog_features.normalise(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(nocar_features))))

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, train_size=0.2, random_state=42)

print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# support vector classifier
svc = LinearSVC()
# training the svc
t=time.time()
svc.fit(X_train, y_train)
print(round(time.time()-t, 4), 'Seconds to train SVC...')

#score of the SVC
svc_score = round(svc.score(X_test, y_test), 8)
print('Test Accuracy of SVC = ', svc_score)


# returns the image with detected bounded box
def drawOnImage(image):

    draw_image = np.copy(image)

    # Rescale data since training data extracted from
    # .png images (scaled 0 to 1 by mpimg) and image we
    # are searching is .jpg (scaled 0 to 255)
    image = img.astype(np.float32)/255

    # Normalise image
    # image = normalise(image)
    # print(image, image.shape)

    mask = np.zeros_like(image[:,:,0])
    vertices = np.array([[(700,400),(1000,720),(1280,720),(1280,400)]])
    mask = cv2.fillPoly(mask, vertices, 1)

    # Get list of windows to search at this stage.
    windows = slidingWindow.slideWindow(image, x_start_stop=[600, 1280], y_start_stop=y_start_stop,
                        xy_window=xy_window_size, xy_overlap=xy_overlap_size, polygon_mask=mask)

    # Return all the windows the classifier has predicted contain car(s) ('positive windows').
    hot_windows = slidingWindow.searchWindows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    # Draw bounding boxes around the windows that the classifier predicted has cars in them
    window_img = slidingWindow.drawBoxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    return window_img

################################################################################
# Test the approach on the test images
################################################################################
im = glob.glob('test_images/test*.jpg')

xy_window_size = (64, 64)
xy_overlap_size = (0.5, 0.5)
y_start_stop = [400, 720] # Min and max in y to search in slide_window()

count=0
for fname in im:
    img = mpimg.imread(fname)
    window_img=drawOnImage(img)
    count+=1
    # Plot image with bounding boxes drawn.
    plt.clf()
    plt.title('vehicle detected and drawn bounding box')
    plt.imshow(window_img)
    plt.savefig("output_images/test" + str(count) + ".jpg")

# pipeline
boundingbox_list = []

def addHeat(heatmap, box_list):
    """Returns `heatmap` with bounding boxes in `bbox_list` added to it.
    `bbox_list` is an array of boxes.

    This function should be applied per frame.
    """
    # Iterate through list of bboxes
    for box in box_list:
        # Add += 1 for all pixels inside each bbox
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def applyThreshold(heatmap, threshold):
    """Returns heatmap with false positives removed."""
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def addBoundingBoxes(input_image, windows, classifier, scaler, draw=True, bboxes_only=False):
    """Adds bounding boxes from `input_image` (one array of arrays) to
    the global variable `bboxes_list`'.
    If `draw` is True, returns image overlaid with bounding boxes.
    """
    global boundingbox_list

    # Normalise image
    # input_image = normalise(input_image)

    hot_windows = slidingWindow.searchWindows(input_image, windows, classifier, scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
    #print("Hot windows: ", hot_windows)
    boundingbox_list.append(hot_windows)
    if draw == True:
        draw_image = np.copy(input_image)
        window_img = draw_boxes(input_image, hot_windows, color=(0, 0, 255), thick=6)
        return window_img


def drawFilteredBoxes(image, all_bboxes, recent_frames_used=20, threshold=5):
    """`all_bboxes` is an array of arrays of bboxes.
    Each element represents a frame. Each element is an array of bboxes found in
    that frame."""

    mask = np.zeros_like(image[:,:,0])
    vertices = np.array([[(700,400),(1000,720),(1280,720),(1280,400)]])
    mask = cv2.fillPoly(mask, vertices, 1)

    # Get list of windows to search at this stage.
    windows = slidingWindow.slideWindow(image, x_start_stop=[600, 1280], y_start_stop=y_start_stop,
                        xy_window=xy_window_size, xy_overlap=xy_overlap_size, polygon_mask=mask)

    # Add bounding boxes from this frame
    addBoundingBoxes(image, windows, svc, X_scaler, draw=False)

    # Adjust parameters if needed
    if len(all_bboxes) < recent_frames_used + 1:
        recent_frames_used = len(all_bboxes) - 1

    # Prepare heatmap template
    frame_heatmap = np.zeros_like(image[:,:,0])

    # Construct heatmap
    for boxlist in all_bboxes[-recent_frames_used:]:
        frame_heatmap = addHeat(frame_heatmap, boxlist)

    # Apply threshold
    frame_heatmap = applyThreshold(frame_heatmap, threshold)

    # Label regions
    labels = label(frame_heatmap)

    # Draw bounding boxes around labelled regions
    draw_img = drawLabeledBoundingBoxes(np.copy(image), labels)

    plt.imshow(draw_img)
    return draw_img

def drawLabeledBoundingBoxes(img, labels):
    """Return image with bounding boxes drawn around the labelled regions.
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def pipeLine(input_image):
    global boundingbox_list
    return drawFilteredBoxes(input_image, boundingbox_list)

count=0
for fname in im:
    img = mpimg.imread(fname)
    window_img=pipeLine(img)
    count+=1
    # Plot image with bounding boxes drawn.
    plt.clf()
    plt.title('vehicle detected and drawn bounding box')
    plt.imshow(window_img)
    plt.savefig("output_images/test_now" + str(count) + ".jpg")
