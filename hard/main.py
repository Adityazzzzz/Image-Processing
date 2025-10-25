import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def load_base():
    global base_path
    base_path = filedialog.askopenfilename()
    lbl_base.config(text=f"Base: {base_path.split('/')[-1]}")

def load_target():
    global target_path
    target_path = filedialog.askopenfilename()
    lbl_target.config(text=f"Target: {target_path.split('/')[-1]}")

def register_images():
    base = cv2.imread(base_path, 0)
    target = cv2.imread(target_path, 0)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(base, None)
    kp2, des2 = orb.detectAndCompute(target, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    h, w = base.shape
    registered = cv2.warpPerspective(target, H, (w, h))

    blended = cv2.addWeighted(base, 0.5, registered, 0.5, 0)
    cv2.imwrite("registered_output.jpg", registered)

    img = Image.fromarray(blended)
    img = img.resize((400, 400))
    imgTk = ImageTk.PhotoImage(img)
    lbl_result.config(image=imgTk)
    lbl_result.image = imgTk
    lbl_status.config(text="Registration Done! Saved as registered_output.jpg")

# GUI
root = Tk()
root.title("Image Registration Tool")
root.geometry("700x600")

Label(root, text="Image Registration using ORB + Homography", font=("Arial", 14, "bold")).pack(pady=10)

Button(root, text="Load Base Image", command=load_base).pack()
lbl_base = Label(root, text="No base image selected")
lbl_base.pack()

Button(root, text="Load Target Image", command=load_target).pack()
lbl_target = Label(root, text="No target image selected")
lbl_target.pack()

Button(root, text="Register Images", command=register_images, bg="#6C63FF", fg="white").pack(pady=10)
lbl_status = Label(root, text="")
lbl_status.pack()

lbl_result = Label(root)
lbl_result.pack(pady=10)

root.mainloop()
