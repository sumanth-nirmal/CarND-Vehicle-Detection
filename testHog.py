#!/usr/bin/python
## Author: sumanth
## Date: March 10,2017
# test script to fetch the hog features

import hog
import glob

cars = glob.glob('vehicles/*/*/*.png')
nocars = glob.glob('non-vehicles/*/*/*.png')

#test image
sample_car = [cars[150]]
sample_nocar = [nocars[150]]

# Plot HOG visualisation
hog.extractFeatures(sample_car, 'YCrCb', (16, 16), 24, 5, 8, 5, 'ALL', False, False, True, \
                        viz=True, viz_only=False, viz_title="viz", hog_viz_name='hogviz')
