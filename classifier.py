#!/usr/bin/python
## Author: sumanth
## Date: March 10,2017
# trains a simple support vector classifier

import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import hog_features


def trainClassifier(color_space, spatial_size, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):

    # read the data
    cars = glob.glob('vehicles/*/*/*.png')
    nocars = glob.glob('non-vehicles/*/*/*.png')
    print(len(cars))
    print(len(nocars))

    # split size
    split_size=0.2 # 20%

    car_features = hog_features.extractFeatures(cars, color_space=color_space,           \
                        spatial_size=spatial_size, hist_bins=hist_bins,     \
                        orient=orient, pix_per_cell=pix_per_cell,           \
                        cell_per_block=cell_per_block,                      \
                        hog_channel=hog_channel, spatial_feat=spatial_feat, \
                        hist_feat=hist_feat, hog_feat=hog_feat)

    nocars_features = hog_features.extractFeatures(nocars, color_space=color_space,      \
                        spatial_size=spatial_size, hist_bins=hist_bins,     \
                        orient=orient, pix_per_cell=pix_per_cell,           \
                        cell_per_block=cell_per_block,                      \
                        hog_channel=hog_channel, spatial_feat=spatial_feat, \
                        hist_feat=hist_feat, hog_feat=hog_feat)

    print(len(car_features))
    print(len(nocars_features))
    # Shuffles features
    car_features = np.array(car_features)
    nocars_features = np.array(nocars_features)
    np.random.shuffle(nocars_features)
    np.random.shuffle(car_features)

    # stack the features
    X = np.vstack((car_features, nocars_features)).astype(np.float64)

    # Fits a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Applies the scaler to X
    scaled_X = X_scaler.transform(X)

    X = hog_features.normalise(X)

    # Defines the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(nocars_features))))

    # Splits up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, train_size=0.2, random_state=42) #rand_state)

    # X_train = np.vstack([X_train, supplement_features])
    # y_train = np.append(y_train, np.ones(len(supplement_features)))

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train))
    print('class: ',len(y_train))

    svc = SVC(probability=True)

    # Checks the training time for the SVC
    t = time.time()

    # train the model
    svc.fit(X_train, y_train)
    print(round(time.time() - t, 2), 'Seconds to train SVC...')

    # Checks the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    print('saving model as pickle')
    # saves the model to disk
    write_obj = {"X_scaler": X_scaler, "svc": svc}
    with open('model.p', 'wb') as file:
        pickle.dump(write_obj, file)
