import cv2
import numpy as np
import matplotlib.pyplot as plt

# load imgs in grayscale mode (0 = grayscale)
base = cv2.imread('img_reg_base.png', 0)
target = cv2.imread('img_reg_target.png', 0)

print("select 3 corresponding points on base img")
plt.imshow(base, cmap='gray')
pts_base = np.array(plt.ginput(3, timeout=0)) 

plt.close()

print("select 3 corresponding points on target img")
plt.imshow(target, cmap='gray')
pts_target = np.array(plt.ginput(3, timeout=0))
plt.close()

# compute affine transform
M = cv2.getAffineTransform(np.float32(pts_target), np.float32(pts_base)) 
# takes 3 points from src and dest
# and computes a 2x3 transformation matrix 'M'

rows, cols = base.shape
registered = cv2.warpAffine(target, M, (cols, rows)) # warping target img to align it with the base img

plt.subplot(141), plt.imshow(base, cmap='gray'), plt.title('Base')
plt.subplot(142), plt.imshow(target, cmap='gray'), plt.title('Target')
plt.subplot(143), plt.imshow(registered, cmap='gray'), plt.title('Registered')

blend = cv2.addWeighted(base, 0.5, registered, 0.5, 0)
plt.subplot(144), plt.imshow(blend, cmap='gray'), plt.title("Blended Comparison")

plt.show()