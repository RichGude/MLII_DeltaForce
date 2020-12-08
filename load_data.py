########################################
# Import libraries
########################################

import cv2                  # for image importing and manipulation
import os                   # for data importing
import numpy as np          # for array processing
import re                   # for string searching and categorization
from tqdm import tqdm       # for SWEET importing visuals!
import matplotlib.pyplot as plt      # for testing frame visuals

########################################
# Create y_train labels
########################################

print(os.getcwd())

os.chdir("data")
y_file = open("train.txt", "r")

labels = []
for line in y_file:
    stripped_line = line.strip()
    labels.append(stripped_line)

y_file.close()

########################################
# Extract images from train video
########################################

# read in the video
# https://www.geeksforgeeks.org/extract-images-from-video-in-python/

train = cv2.VideoCapture(os.getcwd()+"/train.mp4")

try:
    # creating a folder named data
    if "train" not in os.listdir():
        os.mkdir('train')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')
print(os.listdir())

# frame
currentframe = 0

DATA_DIR = os.getcwd() + "/train/"

while True:      # Change to 'True' when reimplementing frame reading

    # reading from frame
    ret, frame = train.read()
    if ret:

        # if video is still left continue creating images
        name = './train/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
train.release()
cv2.destroyAllWindows()

########################################
# Create numpy array of train set
########################################

DATA_DIR = os.getcwd() + "/train/"
RESIZE_WID = int(640/2)
RESIZE_HGT = int(480/8)        # Height is less important than width for movement vectoring

x_train, y_train, indices = [], [], []

print("creating train array...")
for path in tqdm([f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]):
    # images are imported as (480, 640, 3) shape images.  That's enormous, so resize to 1/16 the size
    x_train.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_WID, RESIZE_HGT)))
    index = re.findall(r'\d+', path)
    y_train.append(labels[int(index[0])])
    indices.append(int(index[0]))

x_train, y_train, indices = np.array(x_train), np.array(y_train), np.array(indices)

indices_sorted = indices.argsort()
x_train = x_train[indices_sorted]
y_train = y_train[indices_sorted]
print(indices[indices_sorted])

# Test that the indices are, indeed sorted:
plt.figure(1, figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.title('First three frames of the Training Data')
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[0])

plt.subplot(2, 3, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[1])

plt.subplot(2, 3, 3)
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[2])

plt.subplot(2, 3, 4)
plt.title('Middle three frames of the Training Data')
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[10000])

plt.subplot(2, 3, 5)
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[10001])

plt.subplot(2, 3, 6)
plt.xticks([])
plt.yticks([])
plt.imshow(x_train[10002])
plt.show()
# Images are, indeed, sequential

print("done with train array")
print("train array shape and labels shape:", x_train.shape, y_train.shape)
np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)
