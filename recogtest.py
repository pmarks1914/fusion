import os
from operator import le
import cv2
import dlib
from time import time
import matplotlib.pyplot as plt
import numpy as np
from imutils import face_utils
from skimage.feature import hog, local_binary_pattern
import mahotas as mh
import joblib
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import imgaug.augmenters as iaa

# Resize descriptors to a consistent length
desired_length = 1000
# Initialize the label variable
img_label = []
feature_descriptors_fused = []


def file_looper_transformer_extract_features_and_train():
    source_data_label = ""
    augmentation_seq = iaa.Sequential([
        # iaa.Affine(rotate=(9, 10)),  # Rotate images by -10 to +10 degrees
        iaa.Multiply((0.8, 1.0)),  # Multiply pixel values by a factor between 0.8 and 1.2
        # iaa.GammaContrast(gamma=(2.0, 2.2)),  # Adjust gamma contrast between 0.8 and 1.2
        # iaa.ElasticTransformation(alpha=50, sigma=5)  # Apply elastic transformations
        # iaa.Fliplr(),  # Flip images horizontally with a 50% chance
        # iaa.Multiply((0.8, 1.2)),  # Multiply pixel values by a factor between 0.8 and 1.2
        # iaa.GammaContrast(),  # Adjust gamma contrast between 0.8 and 1.2
        # iaa.ElasticTransformation(alpha=50, sigma=5)  # Apply elastic transformations
        # iaa.Affine(rotate=(0, 10)),  # Rotate images by -10 to +10 degrees
        # iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur with a sigma between 0 and 1.0
        # iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Add Gaussian noise with a scale of 0 to 0.05*255
    ])
    for dirname, _, filenames in os.walk('./jaffedtrain'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            img_name, img_extention = os.path.splitext(filename)
            expression_label = (os.path.join(dirname, filename)).split(os.sep)[2]
            if source_data_label == '':
                source_data_label = (os.path.join(dirname, filename)).split(os.sep)[1]

            if(img_extention == ".jpg" or img_extention == ".png"):
                image = cv2.imread(os.path.join(dirname, filename))
                # Apply data augmentation
                image = augmentation_seq(image=image)
                
                ext_feature, the_label = face_detection_extraction(image, expression_label)
                feature_descriptors_fused.append(ext_feature)
                img_label.append(expression_label)


    # return feature_descriptors_fused, img_label
    svmc(feature_descriptors_fused, img_label, source_data_label)


def face_detection_extraction(image, image_label):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)
    feature_list = []  # List to store the extracted features
    label_list = []  # List to store the corresponding labels

    for face in faces:
        landmarks = predictor(gray, face)
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        extracted_face = image[y:y+h, x:x+w]
        extracted_face = np.array(extracted_face)

        shape = face_utils.shape_to_np(landmarks)
        # Draw on our image, all the found 68 coordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        weight1 = 0.9  # Weight for descriptors lbp
        weight2 = 0.9  # Weight for descriptors hog
        weight3 = 0.2  # Weight for descriptors orb
        weight4 = 0.2  # Weight for descriptors sift
            
        hog_descriptors = hog_extraction(extracted_face)
        try:  
            # Calculate the Z-scores for each descriptor
            zscore_hog_descriptors = (hog_descriptors - np.mean(hog_descriptors)) / np.std(hog_descriptors)
            # Fusion using weighted sum
            hog_descriptors = weight2 * zscore_hog_descriptors   
        except Exception as e:
            # print(str(e), np.std(hog_descriptors))
            pass    
        hog_descriptors = np.resize(hog_descriptors, (desired_length,))

        lbp_descriptors = lbp_extraction(extracted_face)
        # Calculate the Z-scores for each descriptor
        zscore_lbp_descriptors = (lbp_descriptors - np.mean(lbp_descriptors)) / np.std(lbp_descriptors)
        # Fusion using weighted sum
        lbp_descriptors = weight1 * zscore_lbp_descriptors 
        lbp_descriptors = np.resize(lbp_descriptors, (desired_length,))

        orb_descriptors = orb_extraction(extracted_face)
        # Calculate the Z-scores for each descriptor
        # print(" >>>", np.std(orb_descriptors))
        # print(" <  > ", orb_descriptors - np.mean(orb_descriptors) )
        try:
            zscore_orb_descriptors = (orb_descriptors - np.mean(orb_descriptors)) / np.std(orb_descriptors)
            # Fusion using weighted sum
            orb_descriptors = weight3 * zscore_orb_descriptors 
            # print("orb_descriptors ", orb_descriptors)
        except Exception as e:
            # print(str(e))
            pass
        orb_descriptors = np.resize(orb_descriptors, (desired_length,))

        sift_descriptors = sift_extraction(extracted_face)
        try:
            # Calculate the Z-scores for each descriptor
            zscore_sift_descriptors = (sift_descriptors - np.mean(sift_descriptors)) / np.std(sift_descriptors)
            # Fusion using weighted sum
            sift_descriptors = weight4 * zscore_sift_descriptors   
        except Exception as e:
            # print(str(e))
            pass    
        sift_descriptors = np.resize(sift_descriptors, (desired_length,))

        # fused_features = lbp_descriptors + hog_descriptors + sift_descriptors
        fused_features = hog_descriptors
        if lbp_descriptors is not None:
            fused_features += lbp_descriptors
        if sift_descriptors is not None:
            fused_features += sift_descriptors.astype(np.float64)

        # fused_features = np.concatenate((hog_descriptors, lbp_descriptors, sift_descriptors))

        if len(fused_features) > 0:
            feature_list.append(fused_features)
            label_list.append(image_label)  # Add the label to the list

    return feature_list, label_list

        # print(fused_features)
        
        # Display the extracted faces
        # cv2.imshow("Landmark localization", extracted_face)
        # plt.imshow(pixels)
        # plt.axis('off')
        # plt.show()

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
    disply_feature("HOG", hog_features)
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
    disply_feature("LBP", lbp_features)
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
    disply_feature("SIFT", image_with_keypoints)
    return descriptors

def orb_extraction(extracted_face):
    # Create an ORB object with custom parameters
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    # Resize the extracted face to a larger size for better feature extraction
    extracted_face = cv2.resize(extracted_face, (250, 250))
    # Convert the face image to grayscale
    extracted_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(extracted_face, None)
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(extracted_face, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # disply_feature("ORB", image_with_keypoints)
    return descriptors
    
def disply_feature(features_type, features):
    # cv2.imshow(features_type, features)
    # # plt.imshow( features)
    # plt.axis('off')
    # plt.show()
    pass

def svmc(predicted_labels, ground_truth_labels):
    # Load the trained SVM classifier
    classifier = joblib.load("orb_on_ck_svm_classifier_ck.joblib")

    # Initialize lists to store predicted labels and ground truth labels
    predicted_labels = []
    ground_truth_labels = []

   
    # print(ground_truth_labels)
    # Calculate recognition rate metrics
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(ground_truth_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(ground_truth_labels, predicted_labels, average='macro')

    # Print the recognition rate metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

