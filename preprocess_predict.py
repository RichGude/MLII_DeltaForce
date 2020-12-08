'''
This project file is used to load the numpy information obtained from the load_data.py file and pass the information
through a model architecture and train the model

Input:      two numpy training files, x_train.npy and y_train.npy, and one numpy test file, x_test.npy
Output:     train model
'''

##########################################
# Import necessary libraries:
##########################################
import os  # for data importing
import numpy as np  # for array manipulation
import cv2  # for image importing and manipulation
from tqdm import tqdm  # for SWEET importing visuals!

from keras import Model
from keras.layers import Maximum, Input, Conv2D
from keras.layers.core import Dropout, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

""" Potential Process:
The training data is a video of the dashboard view of a car engaging in interstate and city travel.
The model needs to takes the static images of this video, captured at 20 fps, for 17 minutes (20,400 images)
    and output a speed of the car in mph.

Potential path:
The model needs to identify certain stationary objects, such as trees, lampposts, bushes, buildings, etc (not cars),
  and identify their change in position relative to the car camera and angle.

Identifying objects can be done, simply, with edge detection and tracking a shift in pixel values between frames,
  using a Convolutional Network that takes in 2 images stacked together (width, height, 2 x color channels)
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf

  or with MASK R-CNN and/or Fast R-CNN for object detection and running the process through a recurrent network.  This
  system may work great if identifying only buildings, trees, or lampposts and ignoring cars is possible, and then
  gauging the relative movement of the three with respect to the car is achieved.
  https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/mask_rcnn_heads.py

In order to reduce the dimensionality of the data and improve performance:
- Focus on the outer vertical edges of images.  The objects in the vertical center of the object will move little with
  respect to the angle and placement of the camera at any given speed in comparison to the outer vertical edges.  If
  the images are broken into four vertical slices, the first and last should suffice.  I don't know if separating the
  image slices as separate inputs or concatenating the two slices is desirable.

  Separating the image slices may be preferable and building two models based on the left and right image slices with
  derived speeds, with the maximum speed from either side being the true speed:  There are many cars in the images,
  either on the left or right of the camera, though usually not both.  Except when the cars are parked along the side of
  streets during the city-travel portions of the videos, cars are a terrible metric for gauging the speed of the car as
  they are moving at different speeds relative to the car.  It does not appear that there are often cars on both sides
  of the video at the same time, so the effect cars have on the model may be ignored if the errant derived speed from
  cars is ignored.

- The training images are taken near dusk on a cloudy day.  For this reason, the colors are very muted in the images
  often appearing new a grey scale anyway.  Converting the images to a grey scale may be useful to reduce dimensionality
  without affecting results.
"""


#########################################################################
# Define augmentation and editing functions for images and image pairing:
#########################################################################

# As stated previously, the training video is very dark (captured on a cloudy day near dusk).  Improve the brightness of
#   images in order to better capture shift of pixels
def change_brightness(image, bright_factor=1.2):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    # HSB stands for (Hue, Saturation, Brightness); change the image channels to HSV fro RGB
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Perform brightness augmentation only on the third channel
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * bright_factor

    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb


# As stated previously, the focus of the model should be on the outer edges of the image, so the images should be
#   cropped to only focus on the left and right-most quarters of the image (essentially, the middle half is chopped out)
#   In case the images are not otherwise at the proper image scale, resize the images to the proper size as well
def crop_image(image):
    """
    Preprocess the image via resize and then crop (cut out the middle half of the image)
    input:  image (YYY (y), XXX (x), 3) RGB
    output: image (60, 160, 3) as RGB
    """
    RESIZE_WID = int(640 / 2)
    RESIZE_HGT = int(480 / 8)  # Height is less important than width for movement vectoring
    image = cv2.resize(image, (RESIZE_WID, RESIZE_HGT), interpolation=cv2.INTER_AREA)

    # Return two cropped images, one for the left quarter and one for the right
    image_L, image_R = image[:, :int(320 / 4), :], image[:, int(3 * 320 / 4):, :]

    return image_L, image_R


# In order to identify the speed of the car, the change in position of particular objects, identified by the movement,
#   or flow, of particular pixel values, is necessary:
def opticalFlowDense(image_current, image_next):
    """
    input: a time-series-matched pair of images (image_current and image_next) (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    # initialize an empty hsv image
    hsv = np.zeros(image_current.shape)

    # set saturation from input image
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    flow_mat = None  # computed flow image that has the same size as first image and type CV_32FC2
    image_scale = 0.5  # pyr_scale = 0.5 means a classical pyramid, where each layer is half the size the previous
    nb_images = 1  # number of pyramid layers including the initial image (=1, no extra layers are created)
    win_size = 15  # averaging window size; large value = robustness to image noise, but blurred motion field
    nb_iterations = 2  # number of iterations the algorithm does at each pyramid level
    deg_expansion = 5  # size of the pixel neighborhood used to find polynomial expansion in each pixel
    STD = 1.3  # stdev of the Gaussian that is used to smooth derivatives in a basis for the poly-expansion
    extra = 0  # no flags.

    # obtain dense optical flow parameters using the Gunnar Farneback's algorithm
    # (https://docs.opencv.org/3.4/dc/d6b/group__video__track.html):
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        extra)

    # convert flow from cartesian to polar (i.e., magnitude and direction)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction of pixel flow
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude of pixel flow
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype=np.float32)

    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


# Define an all-encompassing function that takes in the x_train data and prepares it for the model (i.e., divides into
#   train, validation, and testing data sets after crops, augments for brightness, and manipulations into an optical
#   flow-dense image).  The ultimate input into the model is:
def batch_shuffle(x_trn, y_trn, step_size):
    """
    Randomly shuffle pairs of rows in the array inputs, separating each pair into train, test, and validation data
       (gives 70% chance to append to train data, 10% for test data, and 20% for validation data)

    Step size is used to determine the time difference between pairs.  Since the training data has a captured video
       frame rate of 20 frames-per-second (fps), a step size of 1 frame for pairing would means that there is a 0.05s
       difference in the composition of the pair.

       Too small a number may mean that no significant change in the content of the image is seen, and the model
       fluctuates wildly on computed speeds; too large and the flow-dense change may be hard to detect.

       Whatever time difference is used in pairing should be repeated in testing (i.e., if test video is captured at a
       different fps, the step_size during test data preparation should be such that the time step in pairs is the same
       as the training)

    return tuple (train_data, test_data, valid_data) numpy arrays
    """
    # Create a list to represent the index of all possible pairs, then randomly shuffle
    randomized_list = np.arange(len(x_trn) - step_size)
    np.random.shuffle(randomized_list)

    # initialize return arrays
    x_trnL_data = []
    x_trnR_data = []
    y_trn_data = []
    x_vldL_data = []
    x_vldR_data = []
    y_vld_data = []
    x_tstL_data = []
    x_tstR_data = []
    y_tst_data = []

    for i in tqdm(randomized_list):
        # identify the two index values of the time-separated image pairs
        idx1 = i
        idx2 = i + step_size

        # Preprocess each image before including in the pair groups
        x_1L, x_1R = crop_image(change_brightness(x_trn[idx1]))
        x_2L, x_2R = crop_image(change_brightness(x_trn[idx2]))
        y_row1 = float(y_trn[idx1])
        y_row2 = float(y_trn[idx2])

        # Create flow-dense image of the image pair and average the two speeds from each
        fdL_image = opticalFlowDense(x_1L, x_2L)
        fdR_image = opticalFlowDense(x_1R, x_2R)
        conv_y = np.mean([y_row1, y_row2])

        # Randomly select one of three bins for pairs (20% chance for valid, 10% for test, 70% for train)
        randInt = np.random.randint(10)
        if 0 <= randInt <= 1:
            x_vldL_data += [fdL_image]
            x_vldR_data += [fdR_image]
            y_vld_data.append(conv_y)

        if randInt == 2:
            x_tstL_data += [fdL_image]
            x_tstR_data += [fdR_image]
            y_tst_data.append(conv_y)

        if randInt > 2:
            x_trnL_data += [fdL_image]
            x_trnR_data += [fdR_image]
            y_trn_data.append(conv_y)

    return x_trnL_data, x_trnR_data, y_trn_data, x_tstL_data, x_tstR_data, y_tst_data, x_vldL_data, x_vldR_data, y_vld_data


##########################################
# Start by uploading image and speed data:
##########################################

# The training data is broken up into two files, x_train and y_train.  From these files, we need to make
#   two working DataFrames that will be fed to separate Keras models:

cwd = os.getcwd()

# Import numpy file
x_train_pre = np.load("./data/x_train.npy")
y_train_pre = np.load("./data/y_train.npy")

# Augment data:
STEP_SIZE = 5  # Step size of 5 equates to a 0.25 s gap between images
x_trainL, x_trainR, y_train, x_testL, x_testR, y_test, x_validL, x_validR, y_valid = batch_shuffle(x_train_pre,
                                                                                                   y_train_pre,
                                                                                                   STEP_SIZE)
np.save("x_testL.npy", x_testL);
np.save("x_testR.npy", x_testR);
np.save("y_test.npy", y_test)
# Verify data size
print('Training data size =\t', np.asarray(x_trainL).shape)
print('Validation data size =\t', np.asarray(x_validL).shape)
print('Test data size =\t\t', np.asarray(x_testL).shape)