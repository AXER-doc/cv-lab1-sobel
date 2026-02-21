import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import requests


# Функция для загрузки тестового изображения

def load_test_image(filename: str) -> np.array:
    with open(filename, "rb") as stream:
        img = Image.open(stream)
        gray = np.array(img.convert('L'))
        return gray.astype(np.float32)


# Собелевские ядра

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


# Нативная реализация с convolve2d (SciPy)

def sobel_native(img):
    gx = convolve2d(img, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(img, sobel_y, mode='same', boundary='symm')
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    return magnitude


# OpenCV реализация

import cv2
from sys import argv

def sobel_opencv(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return sobel_combined

image = load_test_image(argv[1])

print(sobel_native(image))
print(sobel_opencv(image))
