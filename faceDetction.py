import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
import os 
 


# imag detection model 
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# image recognation
recognizer = cv.face.LBPHFaceRecognizer_create()

def resize(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    newDimension = (width, height)
    return cv.resize(frame, newDimension, interpolation=cv.INTER_AREA)

vidio = cv.VideoCapture(0)
while True :
    ret , frame = vidio.read()
    if not ret :
        break
    newframe = resize(frame)
    gray = cv.cvtColor(newframe, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
       cv.rectangle(newframe, (x, y), (x+w, y+h), (255, 0, 0), 2)
       cv.putText(newframe, "youssef", (x-1, y-1), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
       cv.imshow("frame" , newframe)

    if cv.waitKey(25) & 0xFF == ord('q'):
         break




