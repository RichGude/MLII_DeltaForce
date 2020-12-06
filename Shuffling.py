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

x_train = np.load("./data/x_train.npy")
y_train = np.load("./data/y_train.npy")
x_test = np.load("./data/x_train.npy")




def batch_shuffle(x_train, y_train):
    """
    Randomly shuffle pairs of rows in the dataframe, separates train and validation data
    generates a uniform random variable 0->9, gives 20% chance to append to valid data, otherwise train_data
    return tuple (train_data, valid_data) dataframes
    """
    randomized_list = np.arange(len(x_train) - 1)
    np.random.shuffle(randomized_list)

    x_train_data = []
    y_train_data = []
    x_valid_data = []
    y_valid_data = []
    x_test_data = []
    y_test_data = []

    for i in randomized_list:
        idx1 = i
        idx2 = i + 1

        x_row1 = x_train[idx1]
        x_row2 = x_train[idx2]
        y_row1 = y_train[idx1]
        y_row2 = y_train[idx2]

        randInt = np.random.randint(10)
        if 0 <= randInt <= 1:
            x_valid_data += [x_row1]
            x_valid_data += [x_row2]
            y_valid_data.append(y_row1)
            y_valid_data.append(y_row2)

        if randInt == 2:
            x_test_data += [x_row1]
            x_test_data += [x_row2]
            y_test_data.append(y_row1)
            y_test_data.append(y_row2)

        if randInt > 2:
            x_train_data += [x_row1]
            x_train_data += [x_row2]
            y_train_data.append(y_row1)
            y_train_data.append(y_row2)
    return x_valid_data, y_valid_data, x_test_data, y_test_data, x_train_data, y_train_data


# create training and validation set
x_valid_data, y_valid_data, x_test_data, y_test_data, x_train_data, y_train_data = batch_shuffle(x_train, y_train)


print(np.asarray(x_valid_data).shape)
print(len(y_valid_data))
print(np.asarray(x_test_data).shape)
print(np.asarray(y_test_data).shape)
print(np.asarray(x_train_data).shape)
print(np.asarray(y_train_data).shape)

# convert all strings in a list to integers
y_train_data = list(map(float, y_train_data))
y_valid_data = list(map(float, y_valid_data))
y_test_data = list(map(float, y_test_data))





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

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros(image_current.shape)
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype=np.float32)

    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow

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

def preprocess_image_valid_from_path(image_path, scale_factor=0.5):
    img = cv2.resize(image_path, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img, scale_factor)
    return img

def preprocess_image_from_path(image_path, scale_factor=0.5, bright_factor=1):
    img = cv2.resize(image_path, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, bright_factor)
    img = crop_image(img, scale_factor)
    return img


img = preprocess_image_from_path(x_train_data[1])

def generate_training_data(x_train, y_train, batch_size=16, scale_factor=0.5):
    # sample an image from the data to compute image size
    img = preprocess_image_from_path(x_train[1], scale_factor)

    # create empty batches
    image_batch = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]))
    label_batch = np.zeros(batch_size)
    i = 0

    while True:
        speed1 = y_train[i]
        speed2 = y_train[i + 1]

        bright_factor = 0.2 + np.random.uniform()
        img1 = preprocess_image_from_path(x_train[i], scale_factor, bright_factor)
        img2 = preprocess_image_from_path(x_train[i + 1], scale_factor, bright_factor)

        rgb_flow_diff = opticalFlowDense(img1, img2)
        avg_speed = np.mean([speed1, speed2])

        image_batch[int((i / 2) % batch_size)] = rgb_flow_diff
        label_batch[int((i / 2) % batch_size)] = avg_speed

        if not (((i / 2) + 1) % batch_size):
            yield image_batch, label_batch
        i += 2
        i = i % np.asarray(x_train).shape[0]


def generate_validation_data(x_val, y_val, batch_size=16, scale_factor=0.5):
    i = 0
    while i < len(y_val):
        speed1 = y_val[i]
        speed2 = y_val[i + 1]

        img1 = preprocess_image_from_path(x_val[i], scale_factor)
        img2 = preprocess_image_from_path(x_val[i + 1], scale_factor)

        rgb_diff = opticalFlowDense(img1, img2)
        rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
        avg_speed = np.array([[np.mean([speed1, speed2])]])

        yield rgb_diff, avg_speed


def generate_test_data(x_test, y_test, batch_size=16, scale_factor=0.5):
    i = 0
    while i < len(y_test):
        speed1 = y_test[i]
        speed2 = y_test[i + 1]

        img1 = preprocess_image_from_path(x_test[i], scale_factor)
        img2 = preprocess_image_from_path(x_test[i + 1], scale_factor)

        rgb_diff = opticalFlowDense(img1, img2)
        rgb_diff = rgb_diff.reshape(1, rgb_diff.shape[0], rgb_diff.shape[1], rgb_diff.shape[2])
        avg_speed = np.array([[np.mean([speed1, speed2])]])

        yield rgb_diff, avg_speed


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
# import keras.backend.tensorflow_backend as KTF

N_img_height = 66
N_img_width = 220
N_img_channels = 3


def nvidia_model():
    inputShape = (N_img_height, N_img_width, N_img_channels)

    model = Sequential()
    # normalization
    # perform custom normalization before lambda layer in network
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=inputShape))

    model.add(Convolution2D(24, (5, 5),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv1'))

    model.add(ELU())
    model.add(Convolution2D(36, (5, 5),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv2'))

    model.add(ELU())
    model.add(Convolution2D(48, (5, 5),
                            strides=(2, 2),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv3'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, (3, 3),
                            strides=(1, 1),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv4'))

    model.add(ELU())
    model.add(Convolution2D(64, (3, 3),
                            strides=(1, 1),
                            padding='valid',
                            kernel_initializer='he_normal',
                            name='conv5'))

    model.add(Flatten(name='flatten'))
    model.add(ELU())
    model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
    model.add(ELU())
    model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
    model.add(ELU())
    model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
    model.add(ELU())

    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(Dense(1, name='output', kernel_initializer='he_normal'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')

    return model


val_size = len(y_valid_data)
valid_generator = generate_validation_data(x_valid_data, y_valid_data)
BATCH = 16
print()


from keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = 'model-weights-shuffled.h5'
# earlyStopping = EarlyStopping(monitor='val_loss',
#                               patience=1,
#                               verbose=1,
#                               min_delta = 0.23,
#                               mode='min',)
modelCheckpoint = ModelCheckpoint(filepath,
                                  monitor = 'val_loss',
                                  save_best_only = True,
                                  mode = 'min',
                                  verbose = 1,
                                 save_weights_only = True)
callbacks_list = [modelCheckpoint]


model = nvidia_model()
train_size = len(y_train_data)
train_generator = generate_training_data(x_train_data, y_train_data, BATCH)
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 400,
        epochs = 2,
        callbacks = callbacks_list,
        verbose = 1,
        validation_data = valid_generator,
        validation_steps = val_size)

print(history)

print(history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model-v2test mean squared error loss 25 epochs')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
plt.savefig('loss_over_training.png')
plt.show()

model.save('model_shuffled.h5')
