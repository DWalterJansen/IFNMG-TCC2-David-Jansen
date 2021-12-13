import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import skimage.measure

shawls = imread('images/<image_mafa_k-means_seg>.jpg')
plt.figure(num=None, figsize=(8, 6), dpi=80)
imshow(shawls)
entropy = skimage.measure.shannon_entropy(shawls)
print(entropy)
plt.show()