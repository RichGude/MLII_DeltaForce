import cv2
import os
import numpy as np
import re

########################################
# Create y_train labels
########################################

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

while (True):

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
# Extract images from test video
########################################

# read in the video
# https://www.geeksforgeeks.org/extract-images-from-video-in-python/

train = cv2.VideoCapture(os.getcwd()+"/test.mp4")

try:
    # creating a folder named data
    if "test" not in os.listdir():
        os.mkdir('test')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')
print(os.listdir())

# frame
currentframe = 0

DATA_DIR = os.getcwd() + "/test/"

while (True):

    # reading from frame
    ret, frame = train.read()
    if ret:

        # if video is still left continue creating images
        name = './test/frame' + str(currentframe) + '.jpg'
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
RESIZE_TO = 256

x_train, y_train, indices = [], [], []

print("creating train array...")
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x_train.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    index = re.findall(r'\d+', path)
    y_train.append(labels[int(index[0])])
    indices.append(int(index[0]))
    #print("Image", label.split('\n'))
    #plt.imshow(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    #plt.show()


x_train, y_train, indices = np.array(x_train), np.array(y_train), np.array(indices)

indices_sorted = indices.argsort()
x_train = x_train[indices_sorted]
y_train = y_train[indices_sorted]
print(indices[indices_sorted])

print("done with train array")
print("train array shape and labels shape:", x_train.shape, y_train.shape)
np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)

########################################
# Create numpy array of test set
########################################

DATA_DIR = os.getcwd() + "/test/"
RESIZE_TO = 256

x_test, indices = [], []
print("creating test array...")
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]:
    x_test.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    index = re.findall(r'\d+', path)
    indices.append(int(index[0]))
    #print("Image", label.split('\n'))
    #plt.imshow(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    #plt.show()


x_test, indices = np.array(x_test), np.array(indices)
indices_sorted = indices.argsort()
x_test = x_test[indices_sorted]
print(indices[indices_sorted])

print("done with test array")
print("test array shape:", x_test.shape)
np.save("x_test.npy", x_test)
