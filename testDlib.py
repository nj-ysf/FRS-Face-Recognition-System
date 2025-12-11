import cv2
import dlib
import pickle
import numpy as np
import cv2 as cv 

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat")
embedder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load trained classifier
with open("face_recognizer.pkl", "rb") as f:
    clf = pickle.load(f)


def resize(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
vedio = cv.VideoCapture(0)
while True  :
    ret , frame  = vedio.read()
    if not ret :
        break
    frame = resize(frame , 0.50)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for rect in faces:
        shape = predictor(gray, rect)
        face_descriptor = embedder.compute_face_descriptor(frame, shape)
        pred = clf.predict([np.array(face_descriptor)])
        
        # Draw rectangle and label
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, pred[0], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        print(pred[0])

    # cv2.imshow("Result", frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
         cv2.destroyAllWindows()
         break


