# Manual Image Registration (Affine Transform)

## Objective
To manually align two related images using affine transformation that performs translation, rotation, scaling, and shearing.

## Folder Structure
Easy_Manual_Registration/
│
├── main.py
├── base.jpg
├── target.jpg
└── README.md


## Setup Instructions

### Prerequisites
- Python 3.8 or above
- Required Libraries:
pip install opencv-python matplotlib numpy

## How to Run
1. Open a terminal in the project directory

2. Run the project:
python main.py

3. Follow on-screen instructions:
   - Select 3 corresponding points on the base image
   - Select 3 corresponding points on the target image

4. The result window will show:
   - Base image
   - Target image
   - Registered image
   - Blended comparison

## Code Explanation (Core Steps)
### Load Images
cv2.imread('base.jpg', 0)

### Select Points
plt.ginput(3)

### Compute Transformation
cv2.getAffineTransform(pts_target, pts_base)

### Apply Warp
cv2.warpAffine(target, M, (cols, rows))

### Visualize Result
plt.imshow()


## Test Cases

| Case | Description | Expected Output |
|------|-------------|-----------------|
| 1 | Slight rotation/shift | Image aligns with base |
| 2 | Scaling | Scaled target matches base |
| 3 | Identical images | No visible difference |
| 4 | Major distortion | Partial/incorrect alignment |

## Summary

- **Concept**: Manual Image Registration (Affine)
- **Input**: Two grayscale images
- **Output**: Registered image aligned with the base
- **Libraries Used**: OpenCV, NumPy, Matplotlib

## What is Affine Transformation?

An affine transformation is a geometric transformation that preserves:
- Lines remain lines
- Parallel lines remain parallel
- Ratios of distances along lines

It can perform:
- Translation (shifting)
- Rotation
- Scaling
- Shearing

The transformation requires a minimum of 3 point correspondences between the source and destination images.

## Applications

- Medical image alignment
- Document scanning correction
- Satellite image registration
- Computer vision preprocessing
