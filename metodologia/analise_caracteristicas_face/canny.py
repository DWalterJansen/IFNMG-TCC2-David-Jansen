import cv2

filename = 'images/<imagem_mafa>.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image Gray', gray)

img_blur = cv2.GaussianBlur(gray, (3,3), 0)
edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200) # Canny Edge Detection

cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey()
cv2.destroyAllWindows()