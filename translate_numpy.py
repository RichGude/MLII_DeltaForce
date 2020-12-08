import cv2
import os
import numpy as np
import re


x_train, y_train, x_test = np.load("./data/x_train.npy"), np.load("./data/y_train.npy"), np.load("./data/x_test.npy")
y_train = np.array([float(j) for j in y_train])
print('hey')