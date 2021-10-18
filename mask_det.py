import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the mask detection model
mask_model = load_model('mask_detector_model')

# To capture video from webcam.
#cap = cv2.VideoCapture(0)
# To use a video file as input
cap = cv2.VideoCapture('film.mov')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Classify if the face contains a mask or not
        mask_im = img.copy()
        #mask_im = img_to_array(mask_im)
        #mask_im = cv2.cvtColor(mask_im, cv2.COLOR_BGR2RGB)
        mask_im = cv2.resize(mask_im, (224, 224))
        mask_im = preprocess_input(mask_im)
        mask_im = np.expand_dims(mask_im, axis=0)
        (mask, no_mask) = mask_model.predict(mask_im)[0]

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > no_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, no_mask) * 100)

        img = cv2.putText(img=img, text=label, org=(x, y), color=color, fontFace=0, fontScale=1,
                          thickness=2)

    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
cap.release()