import cv2
import dlib
import pickle
import numpy as np

# =======================
# LOAD MODELS
# =======================
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
embedder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load trained classifier
with open("face_recognizer.pkl", "rb") as f:
    clf = pickle.load(f)

# =======================
# UTILS
# =======================
def resize(frame, scale=0.75):
    h = int(frame.shape[0] * scale)
    w = int(frame.shape[1] * scale)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

# =======================
# CAMERA
# =======================
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("❌ Impossible d'ouvrir la caméra")

print("📷 Camera ouverte — appuyez sur 'q' pour quitter")

# =======================
# MAIN LOOP
# =======================
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame (same as train_script.py)
    frame = resize(frame, 0.5)
    
    # Convert to grayscale (same pattern as train_script.py)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # RGB for embedding
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces (same as train_script.py)
    faces = detector(gray)

    for rect in faces:
        # Landmarks
        shape = predictor(gray, rect)

        # Face embedding
        face_descriptor = embedder.compute_face_descriptor(rgb, shape)
        face_descriptor = np.array(face_descriptor).reshape(1, -1)

        # Predict
        proba = clf.predict_proba(face_descriptor)
        confidence = np.max(proba)
        label = clf.classes_[np.argmax(proba)]
        if confidence < 0.6:
            label = "Unknown"

        # Draw bounding box and label
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        print(f"{label} ({confidence*100:.1f}%)")

    # Show frame
    cv2.imshow("Face Recognition - dlib", frame)

    # Quit on 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# =======================
# CLEANUP
# =======================
video.release()
cv2.destroyAllWindows()
print("✅ Programme terminé")
