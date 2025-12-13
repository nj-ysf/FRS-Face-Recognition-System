import cv2 as cv
import os
import numpy as np
import csv
import pandas as pd 
import dlib



def resize(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)


DATA_DIR = "DataSet"
LABEL_FILE = "labels.csv"
MODEL_FILE = "trainer1.yml"
IMG_SIZE = (200, 200)

detctor = dlib.get_frontal_face_detector()
lables   = pd.read_csv(LABEL_FILE)
print(lables)
vidio  = cv.VideoCapture(0)


while True : 
    ret , frame = vidio.read() 
    if not ret :
       break
    frame = resize(frame)
    gray = cv.cvtColor(frame ,cv.COLOR_BGR2GRAY)
    faces = detctor(gray)
    for rect in faces:
        if rect :
            x1, y1 = rect.left(), rect.top()
            x2, y2 = rect.right(), rect.bottom()
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2 )
        

   
    

    
    cv.imshow("face" , frame)
   
    if cv.waitKey(25) & 0xFF == ord('q'):
         break










# -------------------------------
# 1) Load label mappings
# -------------------------------
label_map = {}          # name -> numeric label
reverse_map = {}        # numeric label -> name

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_id = int(row["label_id"])
        name = row["name"]
        label_map[name] = label_id
        reverse_map[label_id] = name

print("Loaded labels:", label_map)

# -------------------------------
# 2) Load all images + labels
# -------------------------------
images = []
labels = []

for name, label_id in label_map.items():
    folder = os.path.join(DATA_DIR, name)
    if not os.path.isdir(folder):
        continue

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # resize to fixed size
        img = cv2.resize(img, IMG_SIZE)

        images.append(img)
        labels.append(label_id)

images = np.array(images)
labels = np.array(labels)

print("Total images loaded:", len(images))

# -------------------------------
# 3) Train/Test Split (80%/20%)
# -------------------------------
split_idx = int(0.8 * len(images))

train_images = images[:split_idx]
train_labels = labels[:split_idx]

test_images = images[split_idx:]
test_labels = labels[split_idx:]

print("Training images:", len(train_images))
print("Testing images:", len(test_images))

# -------------------------------
# 4) Train the LBPH model
# -------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

print("Training the LBPH model... please wait...")
recognizer.train(train_images, train_labels)
recognizer.save(MODEL_FILE)
print("Model saved to:", MODEL_FILE)

# -------------------------------
# 5) Evaluate model on test set
# -------------------------------
correct = 0

for img, true_label in zip(test_images, test_labels):
    predicted_label, confidence = recognizer.predict(img)

    if predicted_label == true_label:
        correct += 1

accuracy = correct / len(test_images) if len(test_images) > 0 else 0

print("Test accuracy:", accuracy * 100, "%")
print("Done.")
