import os
import cv2
import joblib
import dlib
from sklearn import svm
from imutils import face_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import imgaug.augmenters as iaa
from feature_ex import orb_extraction, hog_extraction, lbp_extraction, sift_extraction
from sklearn.impute import SimpleImputer


# Resize descriptors to a consistent length
desired_length = 1000


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

def test_recognition_rate(test_directory):
    # Load the trained SVM classifier
    # classifier = joblib.load("test_3_without_orb_algo_zscore_on_ck_svm_classifier_ck.joblib")
    # classifier = joblib.load("3_without_orb_algo_zscore_on_ck_svm_classifier_ck.joblib")
    # SI model
    classifier = joblib.load("si_3_without_orb_algo_zscore_on_ck_svm_classifier_ckdtrain.joblib")
    # Initialize lists to store predicted labels and ground truth labels
    predicted_labels = []
    ground_truth_labels = []
    # print(test_directory, "filename")
    for dirname, _, filenames in os.walk(str(test_directory)):
        # print(dirname)
        for filename in filenames:
            img_name, img_extention = os.path.splitext(filename)
            print(dirname, img_name, img_extention)
            # print(dirname)
            # Load the test image
            image = cv2.imread(os.path.join(dirname, filename))
            # image = augmentation_seq(image=image)
            # print("img ", image)

            # if(img_extention == ".jpg" or img_extention == ".png"):
            if (img_extention == ".jpg" or img_extention == ".png"):  # Check if the image is successfully loaded
                # Extract features and predict label
                # print(image, img_extention)
                expression_label = (os.path.join(dirname, filename)).split(os.sep)[2]
                features, features_label = face_detection_extraction(image, expression_label)
                # print(features)
                if len(features) > 0:
                    # Create arrays to store the features and labels
                    features = np.array(features)
                    img_labels = np.array(features_label)
    
                    # Reshape the fused features to have two dimensions
                    features = features.reshape(features.shape[0], -1)

                    if(expression_label == "contempt"):
                        # Normalize the feature vectors
                        scaler = StandardScaler()
                        features = scaler.fit_transform(features)

                    # Handle missing values with an imputer transformer
                    imputer = SimpleImputer(strategy='mean')
                    features = imputer.fit_transform(features)

                    # Predict the label using the trained classifier
                    predicted_label = classifier.predict(features)

                    # Append the predicted label and ground truth label
                    predicted_labels.append(predicted_label[0])
                    print("<---->", dirname.split(os.sep)[-1], dirname.split(os.sep) )
                    ground_truth_labels.append(dirname.split(os.sep)[-1])
                else:
                    print("Warning: No features extracted for image", filename)

    # # Calculate recognition rate metrics
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision = precision_score(ground_truth_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(ground_truth_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(ground_truth_labels, predicted_labels, average='macro')

    # Print the recognition rate metrics
    print("Recognition rate")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(ground_truth_labels, predicted_labels)

# Test the recognition rate on the test dataset
test_recognition_rate("./ckdvalidate")
