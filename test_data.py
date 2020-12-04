import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import tensorflow as tf

import os
import cv2
import csv

import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * bright_factor

    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb


def crop_image(image, scale):
    """
    preprocesses the image

    input: image (480 (y), 640 (x), 3) RGB
    output: image (shape is (66, 220, 3) as RGB)

    This stuff is performed on my validation data and my training data
    Process:
             1) Cropping out black spots
             3) resize to (66, 220, 3) if not done so already from perspective transform
    """
    # Crop out sky (top 130px) and the hood of the car (bottom 270px)
    image_cropped = image[130:370, :]  # -> (240, 640, 3)

    height = int(240 * scale)
    width = int(640 * scale)
    image = cv2.resize(image_cropped, (220, 66), interpolation=cv2.INTER_AREA)

    return image


def preprocess_image_from_path(image_path, scale_factor=0.5, bright_factor=1):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, bright_factor)
    img = crop_image(img, scale_factor)
    return img


test_data = np.load('test_data2.npy')
print(test_data.shape)

model2 = tf.keras.models.load_model("model.h5")

prediction = model2.predict(preprocess_image_from_path(test_data))
print(prediction)  # will be a list in a list.