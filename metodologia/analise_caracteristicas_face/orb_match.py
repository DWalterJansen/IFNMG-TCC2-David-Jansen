import numpy as np
import cv2
   
filename_query = 'images/<alguma_imagem_modelo>.jpg'
filename_train = 'images/<imagem_mafa>.jpg'
query_img = cv2.imread(filename_query)
train_img = cv2.imread(filename_train)
   
# Convert it to grayscale
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
query_img_bw = cv2.GaussianBlur(query_img_bw, (3,3), 0)
query_img_bw = cv2.Canny(image=query_img_bw, threshold1=50, threshold2=200) # Canny Edge Detectio


train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
train_img_bw = -train_img_bw
   
# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
   
# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw,None)
  
# Initialize the Matcher for matching
# the keypoints and then match the
# keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors,trainDescriptors)

print(len(matches))
   
# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv2.drawMatches(query_img_bw, queryKeypoints, 
train_img_bw, trainKeypoints, matches[:200],None)
   
final_img = cv2.resize(final_img, (1000,650))
  
# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey()