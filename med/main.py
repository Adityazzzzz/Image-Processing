import cv2
import numpy as np
import matplotlib.pyplot as plt

# load imgs in grayscale mode (0 = grayscale)
base = cv2.imread('base.png', 0)
target = cv2.imread('target.png', 0)

# ORB (Oriented FAST and Rotated BRIEF) is a feature detection and description algorithm
orb = cv2.ORB_create(5000) # create an ORB object with max 5000 features
kp1, des1 = orb.detectAndCompute(base, None)  # Detect keypoints and compute their descriptors for both imgs
kp2, des2 = orb.detectAndCompute(target, None)

# feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # { crossCheck = True } ensures one-to-one matching for each descriptor
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x : x.distance)

# Extract matched keypoints
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#  m.queryIdx and m.trainIdx give indices of matching descriptors in base and target imgs
#  kp1[]/kp2[] access the corresponding keypoints; .pt => (x, y)
#  .reshape(-1, 1, 2) reshapes to (N, 1, 2) — N points; -1 => Let NumPy infer this dimension

# find homography
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
# - Uses RANSAC (Random Sample Consensus) to robustly handle outliers(incorrect matches) in matched kp
# - Returns:
#       H    → 3x3 transformation matrix (used to warp the target img)
#       mask → binary mask indicating which matches are inliers(trusted matches; here, reprojection error < 5.0)

# warp target image
h, w = base.shape
registered = cv2.warpPerspective(target, H, (w, h))

# Prepare images
matched_img = cv2.drawMatches(
    base, 
    kp1, 
    target, 
    kp2, 
    matches[:50], 
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
blended = cv2.addWeighted(base, 0.5, registered, 0.5, 0) # 0 - Scalar added to each pixel after blending | Usually kept at 0 unless brightness adjustment is needed
 
plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(base, cmap='gray'), plt.title('Base Image'), plt.axis('off')
plt.subplot(232), plt.imshow(target, cmap='gray'), plt.title('Target Image'), plt.axis('off')
plt.subplot(233), plt.imshow(registered, cmap='gray'), plt.title('Registered Image'), plt.axis('off')
plt.subplot(234), plt.imshow(matched_img), plt.title('Top 50 ORB Matches'), plt.axis('off')
plt.subplot(235), plt.imshow(blended, cmap='gray'), plt.title('Blended Comparison'), plt.axis('off')

plt.tight_layout()
plt.show()
