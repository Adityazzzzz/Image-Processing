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
    lbl_result.config(image=imgTk, text="", bg="#333") # Show image on dark bg
    lbl_result.image = imgTk
    lbl_status.config(text="Registration Done! Saved as registered_output.jpg")

# ============================================
# GUI (Improved Layout)
# ============================================

# --- Style Constants ---
BG_COLOR = "#F5F7FB"
BTN_COLOR = "#6C63FF"
BTN_FG = "white"
BTN_ACTIVE_BG = "#5A54D4"
TEXT_COLOR = "#333333"
PLACEHOLDER_BG = "#EAEAFB"
PLACEHOLDER_FG = "#555"

HEADER_FONT = ("Segoe UI", 16, "bold")
BODY_FONT = ("Segoe UI", 10)
BTN_FONT = ("Segoe UI", 10, "bold")
STATUS_FONT = ("Segoe UI", 10, "italic")

# --- Root Setup ---
root = Tk()
root.title("Image Registration Tool")
root.minsize(700, 600) # Use minsize for resizability
root.configure(bg=BG_COLOR)

# --- Configure Root Grid (to make result area expand) ---
root.rowconfigure(2, weight=1) # Row 2 (result_frame) will expand
root.columnconfigure(0, weight=1) # Column 0 (main column) will expand

# --- Header ---
Label(root, 
      text="🖼️ Image Registration using ORB + Homography", 
      font=HEADER_FONT, 
      bg=BG_COLOR, 
      fg=TEXT_COLOR
).grid(row=0, column=0, pady=20, sticky="n")

# --- Input Frame (to hold base and target side-by-side) ---
input_frame = Frame(root, bg=BG_COLOR)
input_frame.grid(row=1, column=0, sticky="ew", padx=20)
input_frame.columnconfigure(0, weight=1)
input_frame.columnconfigure(1, weight=1)

# --- Base Image Section ---
base_frame = Frame(input_frame, bg=BG_COLOR)
base_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

Button(base_frame, 
       text="Load Base Image", 
       command=load_base,
       font=BTN_FONT,
       bg=BTN_COLOR,
       fg=BTN_FG,
       activebackground=BTN_ACTIVE_BG,
       activeforeground=BTN_FG,
       relief="flat",
       borderwidth=0,
       pady=5,
       padx=10
).pack(pady=(0, 5))

lbl_base = Label(base_frame, text="No base image selected", bg=BG_COLOR, fg=TEXT_COLOR, font=BODY_FONT)
lbl_base.pack()

# --- Target Image Section ---
target_frame = Frame(input_frame, bg=BG_COLOR)
target_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

Button(target_frame, 
       text="Load Target Image", 
       command=load_target,
       font=BTN_FONT,
       bg=BTN_COLOR,
       fg=BTN_FG,
       activebackground=BTN_ACTIVE_BG,
       activeforeground=BTN_FG,
       relief="flat",
       borderwidth=0,
       pady=5,
       padx=10
).pack(pady=(0, 5))

lbl_target = Label(target_frame, text="No target image selected", bg=BG_COLOR, fg=TEXT_COLOR, font=BODY_FONT)
lbl_target.pack()

# --- Result Frame (this will expand) ---
result_frame = Frame(root, bg=BG_COLOR)
result_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
result_frame.rowconfigure(0, weight=1)
result_frame.columnconfigure(0, weight=1)

# --- Register Button ---
Button(result_frame, 
       text="🔁 Register Images", 
       command=register_images, 
       font=("Segoe UI", 12, "bold"),
       bg=BTN_COLOR, 
       fg=BTN_FG,
       activebackground=BTN_ACTIVE_BG,
       activeforeground=BTN_FG,
       relief="flat",
       borderwidth=0,
       pady=8,
       padx=15
).grid(row=1, column=0, pady=20, sticky="s") # Stick to bottom

# --- Result Image Label ---
lbl_result = Label(result_frame, 
                   text="Registered output will appear here",
                   font=BODY_FONT,
                   bg=PLACEHOLDER_BG,
                   fg=PLACEHOLDER_FG,
                   relief="solid", 
                   borderwidth=1)
lbl_result.grid(row=0, column=0, sticky="nsew")

# --- Status Bar ---
lbl_status = Label(root, text="", bg=BG_COLOR, font=STATUS_FONT)
lbl_status.grid(row=3, column=0, pady=10, sticky="ew")

root.mainloop()