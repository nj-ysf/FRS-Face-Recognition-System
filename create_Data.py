import os
import cv2 as cv

def resize(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

# Folder for saving images
save_folder = "projetDB/DataSet/Brahim"
os.makedirs(save_folder, exist_ok=True)

# Open webcam
video = cv.VideoCapture(0)
i = 0  # start counter


while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to read frame")
        break

    new_frame = resize(frame)

    # Save image
    filename = os.path.join(save_folder, f"Brahim_{i}.jpg")
    cv.imwrite(filename, new_frame)
    print("Saved:", filename)
    i += 1
    if i ==450 :
        break

    # Show frame
    cv.imshow("Capture", new_frame)

    # Press 'q' to quit
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
