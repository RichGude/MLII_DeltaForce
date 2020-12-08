# MLII_DeltaForce
The Delta Force from George Washington's Fall 2020 Machine Learning II class tackles the Comma AI Speed Challenge

The Delta Force Squard consists of:
  Luis Humada
  Sam Cohen
  Rich Gude

The purpose of this project to to analyze a ~3 hour video of dashboard-cam vehicle driving footage with a known driving speed and develop a machine learning model to predict the driving speed of a test set of footage.

The dataset (and link) for the driving footage and known speeds are produced at: https://github.com/commaai/speedchallenge.

# Instructions for how to run the code
If you are interested in running our code, this is the order to run it in:

1). load_data.py
      - This file converts the video files in the data folder to images and creates numpy arrays of the Comma AI footage.

2). load_test.py
      - This file converts Rich's video file from his own excursion in the data folder to images and creates numpy arrays of Rich's footage.
      
3). If you want to retrain the model, run train_model.py. If not, skip this step.

4). preprocess_predict.py
      - This file performs the batch shuffling to create the test file that we use as an input to our predict function. If you want to re-train our model, this step is not necessary
      
5). predict.py
      - This file performs predictions on both the test dataset we created in the preprocess.py and on Rich's video. The Mean Squared Error is spit out, as well as a graph of the accuracy of the predictions.
