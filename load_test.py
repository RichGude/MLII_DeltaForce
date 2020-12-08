'''
This project file is used to load the test video obtained from Rich's frolic with a friend on the streets of Arlington

Input:      one mp4 video file
Output:     three test numpy files (x_richL, x_richR, y_rich)
'''


##########################################
# Import necessary libraries:
##########################################
import os                   # for data importing
import numpy as np          # for array manipulation
import cv2                  # for image importing and manipulation
import re                   # for string searching and categorization
from tqdm import tqdm       # for SWEET importing visuals!
import matplotlib.pyplot as plt         # for visualizing model outputs

#########################################################################
# Define augmentation and editing functions for images and image pairing:
#########################################################################

# As stated previously, the training video is very dark (captured on a cloudy day near dusk).  Improve the brightness of
#   images in order to better capture shift of pixels
def change_brightness(image, bright_factor = 1.2):
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
    image_L, image_R = image[:, :int(320/4), :], image[:, int(3*320/4):, :]

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
    flow_mat = None         # computed flow image that has the same size as first image and type CV_32FC2
    image_scale = 0.5       # pyr_scale = 0.5 means a classical pyramid, where each layer is half the size the previous
    nb_images = 1           # number of pyramid layers including the initial image (=1, no extra layers are created)
    win_size = 15           # averaging window size; large value = robustness to image noise, but blurred motion field
    nb_iterations = 2       # number of iterations the algorithm does at each pyramid level
    deg_expansion = 5       # size of the pixel neighborhood used to find polynomial expansion in each pixel
    STD = 1.3               # stdev of the Gaussian that is used to smooth derivatives in a basis for the poly-expansion
    extra = 0               # no flags.

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
def test_batch_shuffle(x_tst, y_tst, step_size=8):
    """
    Creates pairs of rows in the array inputs and pre-processes all pairs for entry into car model

    Step size is used to determine the time difference between pairs.  Since the training data has a captured video
       frame rate of 20 frames-per-second (fps), a step size of 1 frame for pairing would means that there is a 0.05s
       difference in the composition of the pair.

       Whatever time difference is used in pairing should be repeated in testing (i.e., if test video is captured at a
       different fps, the step_size during test data preparation should be such that the time step in pairs is the same
       as the training).  The model wa made with training data 0.25s between pairs; for 30 fps, this equates to ss=7.5.

    return tuple (train_data, test_data, valid_data) numpy arrays
    """
    # Create a list to represent the index of all possible pairs, then randomly shuffle
    index_list = np.arange(len(x_tst) - step_size)

    # initialize return arrays
    x_tstL_data = []
    x_tstR_data = []
    y_tst_data = []

    for i in tqdm(index_list):
        # identify the two index values of the time-separated image pairs
        idx1 = i
        idx2 = i + step_size

        # Preprocess each image before including in the pair groups
        x_1L, x_1R = crop_image(change_brightness(x_tst[idx1]))
        x_2L, x_2R = crop_image(change_brightness(x_tst[idx2]))
        y_row1 = float(y_tst[idx1])
        y_row2 = float(y_tst[idx2])

        # Create flow-dense image of the image pair and average the two speeds from each
        fdL_image = opticalFlowDense(x_1L, x_2L)
        fdR_image = opticalFlowDense(x_1R, x_2R)
        conv_y = np.mean([y_row1, y_row2])

        x_tstL_data += [fdL_image]
        x_tstR_data += [fdR_image]
        y_tst_data.append(conv_y)

    return x_tstL_data, x_tstR_data, y_tst_data

########################################
# Extract images from test video
########################################

os.chdir("data")

# 'test.txt' is a readout of all of the speeds from the rich car video at 30 fps
y_file = open("test.txt", "r")

labels = []
for line in y_file:
    stripped_line = line.strip()
    labels.append(stripped_line)

y_file.close()
print(labels)

# read in the video
# https://www.geeksforgeeks.org/extract-images-from-video-in-python/
testVid = cv2.VideoCapture(os.getcwd()+"/test_RichCar_video.mp4")
print(testVid)
try:
    # creating a folder named data
    if "rich_test" not in os.listdir():
        os.mkdir('rich_test')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')
print(os.listdir())

# frame
currentframe = 0

DATA_DIR = os.getcwd() + "/rich_test/"

while True:      # Change to 'True' when reimplementing frame reading

    # reading from frame
    ret, frame = testVid.read()
    if ret:

        # if video is still left continue creating images
        name = './rich_test/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
testVid.release()
cv2.destroyAllWindows()

# There are 36 frames for a 2min video.  That means a fps of 30
########################################
# Create numpy array of train set
########################################

DATA_DIR = os.getcwd() + "/rich_test/"
# Set resize to same as training data
RESIZE_WID = int(640/2)
RESIZE_HGT = int(480/8)        # Height is less important than width for movement vectoring

x_rich, y_rich, indices = [], [], []

print("creating train array...")
for path in tqdm([f for f in os.listdir(DATA_DIR) if f[-4:] == ".jpg"]):
    # images are imported as (480, 640, 3) shape images.  That's enormous, so resize to 1/16 the size
    index = re.findall(r'\d+', path)
    if int(index[0]) < 3600:
        x_rich.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_WID, RESIZE_HGT)))
        #print(int(index[0]))
        y_rich.append(labels[int(index[0])])
        indices.append(int(index[0]))

x_rich, y_rich, indices = np.array(x_rich), np.array(y_rich), np.array(indices)

indices_sorted = indices.argsort()
x_rich = x_rich[indices_sorted]
y_rich = y_rich[indices_sorted]

# run data through pre-processing function:
x_richL, x_richR, y_rich = test_batch_shuffle(x_rich, y_rich)
print(x_rich)
print("done with rich_test array")
np.save("x_richL.npy", x_richL); np.save("x_richR.npy", x_richR); np.save("y_rich.npy", y_rich)



