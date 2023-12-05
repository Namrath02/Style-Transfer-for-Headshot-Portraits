import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

img = cv2.imread('./images/face_check.jpg', cv2.IMREAD_GRAYSCALE)
img2 = []
for i in img:
    temp = []
    for j in i:
        if j!= 255:
            j = 255
        else:
            j = 0
        
        temp.append(j)
    img2.append(temp)

plt.imshow(img2,cmap='gray')
plt.show()
# img2 = rgb2gray(img2)
plt.imsave('mask_custom_check.jpg', img2, cmap='gray')

# img2 = np.array(img2)
# img2 = cv2.blur(img2, (4,4))
# plt.imshow(img2, cmap='gray')
# plt.show()


