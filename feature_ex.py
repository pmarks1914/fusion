import os
from operator import le
import cv2
import dlib
import numpy as np
from time import time
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from skimage.feature import hog, local_binary_pattern
import mahotas as mh
import joblib

def orb_extraction(extracted_face):
    # Create an ORB object with custom parameters
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    # Resize the extracted face to a larger size for better feature extraction
    extracted_face = cv2.resize(extracted_face, (230, 230))
    # Convert the face image to grayscale
    extracted_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(extracted_face, None)
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(extracted_face, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # disply_feature("ORB", image_with_keypoints)
    return descriptors

def hog_extraction(extracted_face):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)

    # Define HOG parameters
    orientations = 9  # Number of gradient orientations
    pixels_per_cell = (8, 8)  # Cell size in pixels
    cells_per_block = (2, 2)  # Number of cells in each block

    # Compute HOG features
    hog_image, hog_features = hog(image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)    
    # print(hog_features)
    # disply_feature("HOG", hog_features)
    return hog_features

def lbp_extraction(extracted_face):
    # Compute LBP features

    # Get the dimensions of the image
    # height, width, channels = extracted_face.shape

    # Compute LBP features
    radius = 3
    n_points = 8 * radius
    # convert to 2 dimentional
    extracted_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
    lbp_features = local_binary_pattern(extracted_face, n_points, radius, method='default')
    # print(lbp_features)
    # disply_feature("LBP", lbp_features)
    return lbp_features

# Scale-Invariant Feature Transform descriptor
def sift_extraction(extracted_face):
    # Create a SIFT object
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(extracted_face, None)
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(extracted_face, keypoints, None)
    # print(descriptors)
    # disply_feature("SIFT", image_with_keypoints)
    return descriptors