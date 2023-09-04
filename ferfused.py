# SD FER

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
from sklearn.model_selection import train_test_split, cross_val_score
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
        iaa.Multiply((0.8, 1.0)),  # Multiply pixel values by a factor between 0.8 and 1.2
    ])
    # for dirname, _, filenames in os.walk('./FER/FERPlus2016-Final/FERPlus2016/FER2013Train'):
    for dirname, _, filenames in os.walk('./ed/train_mini_2'):
        for filename in filenames:
            # print("<< ----- >> ", os.path.join(dirname, filename))
            img_name, img_extention = os.path.splitext(filename)
            expression_label = (os.path.join(dirname, filename)).split(os.sep)[3]
            
            if source_data_label == '':
                source_data_label = (os.path.join(dirname, filename)).split(os.sep)[1]
            # print("<< ---expression label--- >> ", expression_label, " < path > ", os.path.join(dirname, filename), " > source_data_label < ", source_data_label )

            if(img_extention == ".jpg" or img_extention == ".jpeg" or img_extention == ".png"):
                image = cv2.imread(os.path.join(dirname, filename))
                # print("<< ----- >> ", os.path.join(dirname, filename))
                # Apply data augmentation 
                # image = augmentation_seq(image=image)
                
                ext_feature, the_label = face_detection_extraction(image, expression_label, img_name, os.path.join(dirname, filename))
                if len(ext_feature) > 0:
                    feature_descriptors_fused.append(ext_feature)
                    img_label.append(expression_label)

    # return feature_descriptors_fused, img_label
    svmc(feature_descriptors_fused, img_label, source_data_label)


def face_detection_extraction(image, image_label, img_name, img_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(f"<<<< gray >>> {image_label}", img_name, "gray", len(gray))

    faces = detector(gray, 1)
    feature_list = []  # List to store the extracted features
    label_list = []  # List to store the corresponding labels
    # if image_label == "happy":
    #     print(f"<<<< gray >>> {image_label}", img_name, len(faces) )

    if len(faces) > 0:
        for face in faces:
            landmarks = predictor(gray, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            extracted_face = image[y:y+h, x:x+w]
            extracted_face = np.array(extracted_face)
            # if image_label == "happy":
            #     print(f"<<<< gray >>> {image_label}", img_name, len(extracted_face[0]) )
            # print("<<<< extracted_face >>>", extracted_face[0], len(extracted_face[0]))

            shape = face_utils.shape_to_np(landmarks)
            # Draw on our image, all the found 68 coordinate points (x,y)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            weight1 = 0.9  # Weight for descriptors lbp
            weight2 = 0.9  # Weight for descriptors hog
            weight3 = 0.2  # Weight for descriptors orb
            weight4 = 0.9  # Weight for descriptors sift

            try:
                # print(" extracted_face ", len(extracted_face) )
                if len(extracted_face[0]) > 0:  
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
                    # print(" hog_descriptors >> ", hog_descriptors)
            
                    lbp_descriptors = lbp_extraction(extracted_face)
                    try:  
                        # Calculate the Z-scores for each descriptor
                        zscore_lbp_descriptors = (lbp_descriptors - np.mean(lbp_descriptors)) / np.std(lbp_descriptors)
                        # Fusion using weighted sum
                        lbp_descriptors = weight1 * zscore_lbp_descriptors 
                    except Exception as e:
                        # print(str(e), np.std(hog_descriptors))
                        pass    
                    lbp_descriptors = np.resize(lbp_descriptors, (desired_length,))
                    # print("lbp_descriptors >>> ", lbp_descriptors)
            
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
                    # print(" sift_descriptors >>> ", sift_descriptors)
            
                    # fused_features = lbp_descriptors + hog_descriptors + sift_descriptors
                    fused_features = []
                    if lbp_descriptors is not None and all(map(lambda element: element is not None, lbp_descriptors)):
                        fused_features = lbp_descriptors.astype(np.float64)

                    if hog_descriptors is not None and all(map(lambda element: element is not None, hog_descriptors)):
                        fused_features = hog_descriptors.astype(np.float64)
                        # print(" fused_features ", image_label, fused_features)
                    if sift_descriptors is not None and all(map(lambda element: element is not None, sift_descriptors)):
                        fused_features += sift_descriptors.astype(np.float64)
                        # print(" fused_features ", img_name, image_label, sift_descriptors.shape)
            
                    # fused_features = np.concatenate((hog_descriptors, lbp_descriptors, sift_descriptors))
                    # if image_label == "happy" and "im0" != img_name and "im1" != img_name and "im2" != img_name and "im3" != img_name:
                    # print(" fused_features ", img_name, image_label, fused_features.shape)
                    if len(fused_features) > 0:
                        feature_list.append(fused_features)
                        label_list.append(image_label)  # Add the label to the list

                else:
                    os.remove(f"{img_path}")
                    
                    print("Warning: No features extracted for image", img_path)
            except Exception as e:
                pass
    else:
        pass

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
    hog_features = None
    try:
        hog_image, hog_features = hog(image_gray, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualize=True)    
        # print(hog_features)
        disply_feature("HOG", hog_features)
    except ValueError as e:
        pass
    return hog_features
    
    
def lbp_extraction(extracted_face):
    # Compute LBP features

    # Get the dimensions of the image
    # height, width, channels = extracted_face.shape

    # Compute LBP features
    radius = 3
    n_points = 8 * radius
    # convert to 2 dimentional
    # print("1 >>>>>>>>> ", extracted_face)
    extracted_face = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2GRAY)
    # print("2 >>>>>>>>> ", extracted_face)
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

# Support Vector Machine Classifier
def svmc(fused_features, img_labels, source_data_label):
    
    print(" fused_features in svm ")
    # Create arrays to store the features and labels
    fused_features = np.array(fused_features)
    # fused_features = np.resize(fused_features, (desired_length,))
    img_labels = np.array(img_labels)
    # img_labels = np.resize(img_labels, (desired_length,))

    
    # Reshape the fused features to have two dimensions
    # fused_features = fused_features.reshape(-1, fused_features.shape[2])
    # print(fused_features.shape)
    # fused_features = fused_features.reshape(1000,)
    # print(fused_features.shape)
    fused_features = fused_features.reshape(fused_features.shape[0], -1)
 
    # Normalize the feature vectors
    # scaler = StandardScaler()
    # fused_features = scaler.fit_transform(fused_features)

    # Handle missing values with an imputer transformer
    imputer = SimpleImputer(strategy='mean')
    fused_features = imputer.fit_transform(fused_features)

    # Split the fused features data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(fused_features, img_labels, test_size=0.014, random_state=41)

    # Create an SVM classifier
    # instantiate classifier with linear kernel and C=100
    svc = svm.SVC(kernel='linear', C=100)

    # Perform 10-fold cross-validation and calculate accuracy
    scores = cross_val_score(svc, fused_features, img_labels, cv=7) 
    # Print accuracy scores for each fold
    for fold, score in enumerate(scores, start=1):
        print(f"Fold {fold}: {score:.2f}")
    # Calculate and print the mean accuracy across all folds
    mean_accuracy = scores.mean()
    print(f"Mean Accuracy: {mean_accuracy:.2f}")

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
    

    print("sd_3_train_without_orb_algo_zscore_on_svm_classifier_ algorithm ", source_data_label)
    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

   
    joblib.dump(svc, f"sd_fer1_3_train_without_orb_algo_zscore_on_svm_classifier_{source_data_label}.joblib")
    # joblib.dump(svc, f"lbp_zscore_on_svm_classifier_{source_data_label}.joblib")
    # joblib.dump(svc, f"hog_on_svm_classifier_{source_data_label}.joblib")
    # joblib.dump(svc, f"sift_on_svm_classifier_{source_data_label}.joblib")

file_looper_transformer_extract_features_and_train()
