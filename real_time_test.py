import cv2 as cv
import os

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = "DataSet"
MODEL_FILE = "trainer.yml"
HAAR_CASCADE = cv.data.haarcascades + "haarcascade_frontalface_default.xml"

# -----------------------------
# Load Haar cascade
# -----------------------------
face_cascade = cv.CascadeClassifier(HAAR_CASCADE)
if face_cascade.empty():
    raise Exception("Cannot load Haar cascade xml file!")

# -----------------------------
# Load LBPH model
# -----------------------------
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

# -----------------------------
# Map labels to names
# -----------------------------
# Automatically read folder names as labels
label_map = {}
for idx, name in enumerate(sorted(os.listdir(DATASET_DIR))):
    label_map[idx] = name

print("Label mapping:", label_map)

# -----------------------------
# Start webcam
# -----------------------------
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Recognize face
        label_id, confidence = recognizer.predict(face_roi)
        name = label_map.get(label_id, "Unknown")

        # Draw rectangle and label
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{name} ({int(confidence)})"
        cv.putText(frame, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
