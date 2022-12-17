import tensorflow as tf
import numpy as np
import cv2
import os

model = tf.keras.models.load_model("oldmainet.h`5")

def conv_img(img):
    # rescale to big to ensure enough pixel data
    height, width = img.shape[:2]
    img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_CUBIC)
    img = cv2.resize(img, (120, 120)) /255.
    return np.array(img).reshape(-1, 120, 120, 1)

def conv_img_to_rgb(gray):
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    gray = cv2.resize(gray, (128, 128))
    print("New Shape -> ",gray.shape)
    return np.array(gray).reshape(-1, 128, 128, 3)

tests = os.listdir("tester")

for img in tests:
    imgg = cv2.imread(os.path.join("tester",img))
    imgg = cv2.resize(imgg, (128, 128))
    res = model.predict(np.array(imgg).reshape(-1, 128, 128, 3))

    result = 'HeatStress'
    if res[0][0] > .50: result = "HeatStress"
    else: result = "Normal"

    print(f"Totest -> {img}, Model Result -> {res}, Identification -> {result}")



