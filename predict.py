'''
This project file is used to load the numpy information obtained from the train_model.py file and pass the information
through a saved model architecture and predict the speed of

Input:      two numpy test files (left and right images) and 'model_car.h5' model file
Output:     predicted speeds
'''

##########################################
# Import necessary libraries:
##########################################
import os                   # for data importing
import numpy as np          # for array manipulation
import cv2                  # for image importing and manipulation
from tqdm import tqdm       # for SWEET importing visuals!
import matplotlib.pyplot as plt         # for visualizing model outputs

import keras
from keras import Model
from keras.layers import Maximum, Input, Conv2D
from keras.layers.core import Dropout, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error


########################################
# Extract images from train video
########################################
cwd = os.getcwd()

# load test data (this data is shuffled dense-flow image pairs from the original training frames)
x_testL = np.load('x_testL.npy')
x_testR = np.load('x_testR.npy')
y_test = np.load('y_test.npy')

# load test data (this data is non-shuffled dense-flow image pairs from the rich car frames)
x_richL = np.load('./data/x_richL.npy')
x_richR = np.load('./data/x_richR.npy')
y_rich = np.load('./data/y_rich.npy')

# To improve the visualization of the prediction error, sort the test arrays by the speed
speed_sorted = y_test.argsort()
x_testL = x_testL[speed_sorted]
x_testR = x_testR[speed_sorted]
y_test = y_test[speed_sorted]

# load and run model previously trained
model = keras.models.load_model("model_car.h5")
# Generate speed predictions values
y_pred = model.predict([x_testL, x_testR])
y_pred_rich = model.predict([x_richL, x_richR])

# Plot the y_pred versus the y_test to see the differences in the modeled outputs
plt.plot(np.arange(len(y_test)), y_test, 'r-', label='True Speed')
plt.scatter(np.arange(len(y_test)), y_pred, c='blue', label='Pred Speed')
plt.xlabel('Frames')
plt.ylabel('Car Speed (mph)')
plt.title('Actual versus Predicted Car Speeds from Training Footage')
plt.legend()
plt.show()
print("Final MSE on test set:", mean_squared_error(y_test, y_pred))

plt.savefig('train_test_results.png')

# Plot the y_pred_rich versus the y_rich to see the differences in the modeled outputs
plt.plot(np.arange(len(y_rich)), y_rich, 'r-', label='True Speed')
plt.scatter(np.arange(len(y_rich)), y_pred_rich, c='blue', label='Pred Speed')
plt.xlabel('Frames')
plt.ylabel('Car Speed (mph)')
plt.title('Actual versus Predicted Car Speeds from Rich Car Footage')
plt.legend()
plt.show()

plt.savefig('rich_test_results.png')
print("Final MSE on Rich's video:", mean_squared_error(y_rich, y_pred_rich))

