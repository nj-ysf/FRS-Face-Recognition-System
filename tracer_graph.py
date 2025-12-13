import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import face_recognition

dataset_path = "DataSet"
hist = []

# Charger modèle de reconnaissance faciale Dlib
face_detector_cnn = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # modèle CNN
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # Conversion et amélioration contraste
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Détecter les visages avec CNN
        faces = face_detector_cnn(image_rgb, 1)  # upsample=1 pour petites images

        if len(faces) == 0:
            print(f"Aucun visage détecté dans {image_name}")
            continue

        # Extraire l'embedding pour chaque visage
        for face in faces:
            # CNN retourne rectangle + score, on prend face.rect
            rect = face.rect
            shape = landmark_predictor(image_rgb, rect)
            encoding = np.array(face_recognition.face_encodings(image_rgb, [(rect.top(), rect.right(), rect.bottom(), rect.left())]))

            if len(encoding) == 0:
                print(f"Landmarks non trouvés dans {image_name}")
                continue

            l2_norm = np.linalg.norm(encoding[0])
            hist.append(l2_norm)

# Tracer le graphe
plt.plot(hist, marker='o')
plt.title("Norme L2 unique pour chaque visage")
plt.xlabel("Index image")
plt.ylabel("L2 Norm")
plt.grid(True)
plt.show()
