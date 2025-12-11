import os
import cv2
import dlib
import pickle
import numpy as np
from sklearn.svm import SVC

# --------------------------
# Paths to models
# --------------------------
landmarks_path = "shape_predictor_68_face_landmarks (1).dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
dataset_path = "DataSet"  # your folder structure: dataset/person_name/image.jpg
model_save_path = "face_recognizer.pkl"

# --------------------------
# Initialize models
# --------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarks_path)
embedder = dlib.face_recognition_model_v1(face_rec_model_path)

# --------------------------
# Prepare training data 
# --------------------------
X = []  # embeddings
y = []  # labels


for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        for rect in faces:
            shape = predictor(gray, rect)
            face_descriptor = embedder.compute_face_descriptor(img, shape)
            X.append(np.array(face_descriptor))
            y.append(person_name)
            break  # only use first face per image

X = np.array(X)
y = np.array(y)

# --------------------------
# Train SVM classifier
# --------------------------
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y)

# --------------------------
# Save classifier
# --------------------------
with open(model_save_path, "wb") as f:
    pickle.dump(clf, f)

print("Training complete! Model saved to", model_save_path)
