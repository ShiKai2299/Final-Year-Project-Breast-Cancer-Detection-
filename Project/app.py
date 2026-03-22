import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense

# =====================
# CONFIGURATION
# =====================
MODEL_PATH = "VGG16.h5"
IMG_SIZE = 256

# =====================
# CUSTOM DENSE FIX
# =====================
class CompatibleDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)

# =====================
# LOAD MODEL
# =====================
def load_vgg16(path):
    return load_model(
        path,
        compile=False,
        custom_objects={"Dense": CompatibleDense}
    )

try:
    model = load_vgg16(MODEL_PATH)
    print("Success: VGG16 model is ready for prediction.")
except Exception as e:
    messagebox.showerror("Critical Error", f"Model could not be loaded: {e}")
    raise SystemExit

# =====================
# IMAGE PREPROCESSING
# =====================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Cannot read image file.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# =====================
# VALIDATION IMAGE BEFORE PREDICTION
# =====================
def is_mammogram(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    b, g, r = cv2.split(img.astype(np.float32))
    rg_diff = np.mean(np.abs(r - g))
    gb_diff = np.mean(np.abs(g - b))

    if rg_diff > 5 or gb_diff > 5:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean_val = np.mean(gray)
    if mean_val < 10 or mean_val > 245:
        return False

    std_val = np.std(gray)
    if std_val < 10:
        return False

    bright_pixels = np.sum(gray > 50)
    total_pixels = gray.size
    bright_ratio = bright_pixels / total_pixels

    if bright_ratio < 0.1:
        return False

    return True

# =====================
# PREDICTION
# =====================
def predict_image(image_path):
    if not is_mammogram(image_path):
        return "Invalid Image Type", 0.0

    img = preprocess_image(image_path)
    prediction = float(model.predict(img, verbose=0)[0][0])

    if prediction >= 0.6:
        return "Malignant", prediction
    elif prediction <= 0.4:
        return "Benign", 1 - prediction
    else:
        return "Uncertain", abs(prediction - 0.5) * 2

# =====================
# TKINTER FUNCTIONS
# =====================
def choose_image():
    global selected_image_path, img_display

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    selected_image_path = file_path

    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_display = ImageTk.PhotoImage(img)
    image_label.config(image=img_display)
    image_label.image = img_display

    result_label.config(text="Result: -", fg="black")
    confidence_label.config(text="Confidence: -")

def check_prediction():
    if not selected_image_path:
        messagebox.showwarning("Warning", "Please select an image first.")
        return

    try:
        result, confidence = predict_image(selected_image_path)
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return

    if result == "Invalid Image Type":
        messagebox.showerror("Invalid Input", "Please upload a valid mammogram image.")
        reset_app()
        return

    if result == "Uncertain":
        result_label.config(text="Result: Uncertain", fg="orange")
        confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")
        return

    result_label.config(text=f"Result: {result}")
    confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")

    if result == "Malignant":
        result_label.config(fg="red")
    else:
        result_label.config(fg="green")

def reset_app():
    global selected_image_path, img_display

    selected_image_path = None
    img_display = None

    image_label.config(image="")
    image_label.image = None
    result_label.config(text="Result: -", fg="black")
    confidence_label.config(text="Confidence: -")

# =====================
# TKINTER UI
# =====================
root = tk.Tk()
root.title("Breast Cancer Detection System")
root.geometry("600x500")
root.resizable(False, False)

selected_image_path = None
img_display = None

title_label = tk.Label(
    root,
    text="Breast Cancer Detection",
    font=("Arial", 18, "bold")
)
title_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

choose_btn = tk.Button(
    btn_frame,
    text="Choose Image",
    width=15,
    command=choose_image
)
choose_btn.grid(row=0, column=0, padx=5)

check_btn = tk.Button(
    btn_frame,
    text="Check",
    width=15,
    command=check_prediction
)
check_btn.grid(row=0, column=1, padx=5)

reset_btn = tk.Button(
    btn_frame,
    text="Refresh",
    width=15,
    command=reset_app
)
reset_btn.grid(row=0, column=2, padx=5)

result_label = tk.Label(
    root,
    text="Result: -",
    font=("Arial", 14)
)
result_label.pack(pady=10)

confidence_label = tk.Label(
    root,
    text="Confidence: -",
    font=("Arial", 12)
)
confidence_label.pack()

close_btn = tk.Button(
    root,
    text="Close",
    width=15,
    command=root.destroy
)
close_btn.pack(pady=20)

root.mainloop()