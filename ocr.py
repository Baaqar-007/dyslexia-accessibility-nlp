import cv2
import numpy as np
from PIL import Image
import os

output_dir = 'extracted_characters'
os.makedirs(output_dir, exist_ok=True)

def extract_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    char_count = 0 
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        char_img = binary_img[y:y+h, x:x+w]

        char_img = cv2.resize(char_img, (28, 28))  

        pil_img = Image.fromarray(char_img)


        char_img_name = os.path.join(output_dir, f'char_{char_count}.png')
        pil_img.save(char_img_name)

        print(f'Saved {char_img_name}')
        char_count += 1


image_path = '2new_test_ocr.jpeg'  
extract_characters(image_path)
