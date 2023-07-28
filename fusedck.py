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

# Resize descriptors to a consistent length
desired_length = 1000
# Initialize the label variable
img_label = []
feature_descriptors_fused = []


def file_looper_transformer_extract_features_and_train():
    source_data_label = ""
    for dirname, _, filenames in os.walk('./jaffe'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            img_name, img_extention = os.path.splitext(filename)
            expression_label = (os.path.join(dirname, filename)).split(os.sep)[2]
            if source_data_label == '':
                source_data_label = (os.path.join(dirname, filename)).split(os.sep)[1]

            if(img_extention == ".jpg" or img_extention == ".png"):
                image = cv2.imread(os.path.join(dirname, filename))
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

        hog_descriptors = hog_extraction(extracted_face)
        hog_descriptors = np.resize(hog_descriptors, (desired_length,))

        lbp_descriptors = lbp_extraction(extracted_face)
        lbp_descriptors = np.resize(lbp_descriptors, (desired_length,))

        orb_descriptors = orb_extraction(extracted_face)
        orb_descriptors = np.resize(orb_descriptors, (desired_length,))

        sift_descriptors = sift_extraction(extracted_face)
        sift_descriptors = np.resize(sift_descriptors, (desired_length,))

        fused_features = lbp_descriptors
        # fused_features = np.concatenate((hog_descriptors, lbp_descriptors, orb_descriptors, sift_descriptors))

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
    # Create an ORB object
    orb = cv2.ORB_create()
    extracted_face = cv2.resize(extracted_face, (100, 100))
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(extracted_face, None)
    # print(descriptors)
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

# Support Vector Machine Classifier
def svmc(fused_features, img_labels, source_data_label):
    # Create arrays to store the features and labels
    fused_features = np.array(fused_features)
    img_labels = np.array(img_labels)
 
    # Reshape the fused features to have two dimensions
    fused_features = fused_features.reshape(fused_features.shape[0], -1)

    # Normalize the feature vectors
    scaler = StandardScaler()
    fused_features = scaler.fit_transform(fused_features)

    # Handle missing values with an imputer transformer
    imputer = SimpleImputer(strategy='mean')
    fused_features = imputer.fit_transform(fused_features)

    # Split the fused features data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(fused_features, img_labels, test_size=0.03, random_state=44)

    # Create an SVM classifier
    # instantiate classifier with linear kernel and C=100
    svc = svm.SVC(kernel='linear', C=100)

    # Train the classifier
    svc.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = svc.predict(X_test)
    # print(y_pred)
    # print("< ---------------- >")
    # print(y_test)
    # print("< ---------------- >")
    # print(X_train)
    # print("< ---------------- >")
    # print(X_test)

    # Evaluate the classifier's performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

   
    joblib.dump(svc, f"svm_classifier_{source_data_label}.joblib")

file_looper_transformer_extract_features_and_train()
