import cv2
import os

base_dir = "DataSet"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("Starting face cropping...\n")

for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Processing: {person}")

    counter = 1
    for img_name in os.listdir(person_dir):
        path = os.path.join(person_dir, img_name)
        
        img = cv2.imread(path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue  # skip images without a face

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            # overwrite old files with clean cropped ones
            save_path = os.path.join(person_dir, f"{counter}.jpg")
            cv2.imwrite(save_path, face)
            counter += 1

    print(f"> Done: {counter-1} faces saved for {person}\n")

print("All face cropping completed successfully!")
