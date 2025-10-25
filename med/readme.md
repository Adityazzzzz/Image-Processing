# Automatic Image Registration (ORB + Homography)

## Overview
This project performs automatic **image registration** — aligning a target image with a base (reference) image — using **feature-based methods**.

It uses **ORB (Oriented FAST and Rotated BRIEF)** for feature detection and description, and **Homography** for perspective correction.

## Features
- Fully automatic registration (no manual point selection)
- Works for rotated, scaled, and perspective-distorted images
- Displays top 50 feature matches and blended overlay
- Produces aligned (registered) output image

## Techniques Used
| Step | Technique | Description |
|------|------------|-------------|
| Feature Detection | ORB | Detects and describes distinctive keypoints |
| Feature Matching | Brute Force Matcher | Finds best feature correspondences |
| Transformation | Homography | Corrects rotation, scale, and perspective |
| Alignment | warpPerspective() | Aligns the target to the base |

## Setup and Usage

### 1. Install dependencies
pip install opencv-python matplotlib numpy

### 2. Folder structure
Medium_FeatureBased_Registration/
│
├── main.py
├── base.png
├── target.png
└── README.md

### 3. Run the project
python main.py

### 4. Output
- Base image
- Target image
- Registered (aligned) image
- ORB feature matches
- Blended overlay for comparison

## Test Cases
- **Rotated Image**: Rotate base image and align automatically
- **Scaled Image**: Zoomed version of base image
- **Perspective View**: Capture from an angle
- **Partial Overlap**: Crop and align subset of scene

## Example Results
| Image | Description |
|-------|-------------|
| Base | Original reference image |
| Target | Transformed input image |
| Registered | Aligned result |
| Matches | Visualization of top 50 feature matches |
| Blended | Overlay comparison |

## Applications
- Medical image alignment
- Satellite image stitching
- Panorama creation
- Augmented reality