import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

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


def capture_frames(video_source, speed_data):
    '''
    Captures .mp4 video frames to .jpg images and creates a .csv to store the capture information
    '''

    num_frames = speed_data.shape[0]

    # create VideoCapture instance
    cap = cv2.VideoCapture(video_source)
    # set frame count
    cap.set(cv2.CAP_PROP_FRAME_COUNT, num_frames)

    with open('/home/ubuntu/Comma/MLII_DeltaForce/data/driving.csv', 'w') as csvfile:
        fieldnames = ['image_path', 'frame', 'speed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(num_frames):
            # set frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # read the frame
            success, image = cap.read()

            if success:
                image_path = os.path.join('/home/ubuntu/Comma/MLII_DeltaForce/data/trainingframes', str(idx) + '.jpg')

                # save image to IMG folder
                cv2.imwrite(image_path, image)

                # write row to driving.csv
                writer.writerow({'image_path': image_path,
                                 'frame': idx,
                                 'speed': speed_data[idx],
                                 })
            else:
                print('Failed to read frame ', idx)

        print('Done!')


# capture_frames('/home/ubuntu/Comma/MLII_DeltaForce/data/train.mp4', np.loadtxt('/home/ubuntu/Comma/MLII_DeltaForce/data/train.txt'))


df = pd.read_csv('/home/ubuntu/Comma/MLII_DeltaForce/data/driving.csv')
print(df.head(10))
print()

video_fps = 20
times = np.asarray(df['frame'], dtype = np.float32) / video_fps
speeds = np.asarray(df['speed'], dtype=np.float32)
plt.plot(times, speeds, 'r-')
plt.title('Speed vs Time')
plt.xlabel('time (secs)')
plt.ylabel('speed (mph)')
plt.show()

print(df.tail(5))


def batch_shuffle(dframe):
    """
    Randomly shuffle pairs of rows in the dataframe, separates train and validation data
    generates a uniform random variable 0->9, gives 20% chance to append to valid data, otherwise train_data
    return tuple (train_data, valid_data) dataframes
    """
    randomized_list = np.arange(len(dframe) - 1)
    np.random.shuffle(randomized_list)

    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for i in randomized_list:
        idx1 = i
        idx2 = i + 1

        row1 = dframe.iloc[[idx1]].reset_index()
        row2 = dframe.iloc[[idx2]].reset_index()

        randInt = np.random.randint(10)
        if 0 <= randInt <= 1:
            valid_frames = [valid_data, row1, row2]
            valid_data = pd.concat(valid_frames, axis=0, join='outer', ignore_index=False)
        if randInt == 2:
            test_frames = [test_data, row1, row2]
            test_data = pd.concat(test_frames, axis=0, join='outer', ignore_index=False)
        if randInt > 2:
            train_frames = [train_data, row1, row2]
            train_data = pd.concat(train_frames, axis=0, join='outer', ignore_index=False)
    return train_data, valid_data, test_data


# create training and validation set
train_data, valid_data, test_data = batch_shuffle(df)

# verify data size
print('Training data size =', train_data.shape)
print('Validation data size =', valid_data.shape)
print('Test data size =', test_data.shape)


x = random.sample(range(1,len(train_data)), 15)

plt.figure(1, figsize=(20,10))
for i,value in enumerate(x):
    img = mpimg.imread(train_data.iloc[value]['image_path'])
    plt.subplot(3,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)


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


img1 = mpimg.imread(train_data.iloc[34]['image_path'])
img2 = mpimg.imread(train_data.iloc[23]['image_path'])

rgb_diff = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) - cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
hsv_diff = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV) - cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
sat = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:,:,1] - cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,1]
inv_sat = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,1]*-1
hue = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)[:,:,0]

plt.figure(1, figsize=(20,10))

plt.subplot(2,3,1)
plt.xticks([])
plt.yticks([])
plt.imshow(img2)

plt.subplot(2,3,2)
plt.xticks([])
plt.yticks([])
plt.imshow(inv_sat)
plt.show()

plt.subplot(2,3,3)
plt.xticks([])
plt.yticks([])
plt.imshow(sat)
plt.show()

plt.subplot(2,3,4)
plt.xticks([])
plt.yticks([])
plt.imshow(hue)
plt.show()

plt.subplot(2,3,5)
plt.xticks([])
plt.yticks([])
plt.imshow(rgb_diff)
plt.show()

plt.subplot(2,3,6)
plt.xticks([])
plt.yticks([])
plt.imshow(hsv_diff)
plt.show()


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

img1 = mpimg.imread(train_data.iloc[0]['image_path'])
img2 = mpimg.imread(train_data.iloc[1]['image_path'])

rgb_diff = opticalFlowDense(img1,img2)
plt.xticks([])
plt.yticks([])
plt.imshow(rgb_diff)
plt.show()


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
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img, scale_factor)
    return img

def preprocess_image_from_path(image_path, scale_factor=0.5, bright_factor=1):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = change_brightness(img, bright_factor)
    img = crop_image(img, scale_factor)
    return img

img = preprocess_image_from_path(train_data.iloc[404]['image_path'])
img_next = preprocess_image_from_path(train_data.iloc[405]['image_path'])

plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(img_next)
plt.show()

rgb_flow = opticalFlowDense(img,img_next)
plt.figure()
plt.imshow(rgb_flow)
plt.show()


def generate_training_data(data, batch_size=16, scale_factor=0.5):
    # sample an image from the data to compute image size
    img = preprocess_image_from_path(train_data.iloc[1]['image_path'], scale_factor)

    # create empty batches
    image_batch = np.zeros((batch_size, img.shape[0], img.shape[1], img.shape[2]))
    label_batch = np.zeros(batch_size)
    i = 0

    while True:
        speed1 = data.iloc[i]['speed']
        speed2 = data.iloc[i + 1]['speed']

        bright_factor = 0.2 + np.random.uniform()
        img1 = preprocess_image_from_path(data.iloc[i]['image_path'], scale_factor, bright_factor)
        img2 = preprocess_image_from_path(data.iloc[i + 1]['image_path'], scale_factor, bright_factor)

        rgb_flow_diff = opticalFlowDense(img1, img2)
        avg_speed = np.mean([speed1, speed2])

        image_batch[int((i / 2) % batch_size)] = rgb_flow_diff
        label_batch[int((i / 2) % batch_size)] = avg_speed

        if not (((i / 2) + 1) % batch_size):
            yield image_batch, label_batch
        i += 2
        i = i % data.shape[0]


def generate_validation_data(data, batch_size=16, scale_factor=0.5):
    i = 0
    while i < len(data):
        speed1 = data.iloc[i]['speed']
        speed2 = data.iloc[i + 1]['speed']

        img1 = preprocess_image_from_path(data.iloc[i]['image_path'], scale_factor)
        img2 = preprocess_image_from_path(data.iloc[i + 1]['image_path'], scale_factor)

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
plt.show()


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


val_size = len(valid_data.index)
valid_generator = generate_validation_data(valid_data)
BATCH = 16
print()


from keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = 'model-weights-Vtest3.h5'
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
train_size = len(train_data.index)
train_generator = generate_training_data(train_data, BATCH)
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 400,
        epochs = 25,
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