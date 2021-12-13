import cv2

# Load the cascade
base_path = "classifiers/"
face_cascade = cv2.CascadeClassifier(base_path + 'haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('images/<name-frame>.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 6)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.imwrite('frame_detect_violajones.jpg', img)
cv2.waitKey()