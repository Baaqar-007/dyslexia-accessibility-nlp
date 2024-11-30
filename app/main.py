from flask import Flask, request, jsonify, render_template, send_file
import os
import threading
import time
import cv2
import numpy as np
from PIL import Image
import shutil
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import datetime

app = Flask(__name__, template_folder='templates')

# ----------------------------- Constants -----------------------------

# Model Paths
MLP_MODEL_PATH = 'models/mlp_model.pkl'
NLP_MODEL_PATH = 'models/sequence_anomaly.h5'
CNN_MODEL_PATH = 'models/pattern_classifier.h5'


# Other Paths
OUTPUT_DIR = "output"
IMAGE_PATH = "uploaded_image.jpg"  # Shared path for uploaded image
PDF_FILE_PATH = "detailed_report.pdf"
TEMP_RESULT = "result_summary.json"  # Temporary file for result summary

# Image Dimensions
IMG_WIDTH, IMG_HEIGHT = 64, 64

# ---------------------------- Load Models ----------------------------

mlp_model = joblib.load(MLP_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)
nlp_model = load_model(NLP_MODEL_PATH)

# ------------------------- Helper Functions --------------------------

def preprocess_for_mlp(img):
    """
    Preprocesses an image for MLP model prediction.
    """
    return img.reshape(1, -1) / 255.0

def preprocess_for_cnn(img):
    """
    Preprocesses an image for CNN model prediction.
    """
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1).astype('float32') / 255.0
    return img

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # print(img_array)
    prediction = model.predict(img_array)
    return 'Reversal' if prediction < 0.8 else 'Normal'

def detect_sequence_anomaly(seq,max_len=5):
    """
    Detects sequence anomalies using the trained NLP model.

    Args:
        seq (str): Input sequence to analyze.
        model (Sequential): Trained NLP model for anomaly detection.
        max_len (int): Maximum sequence length for padding.

    Returns:
        str: Result of anomaly detection ("Anomaly detected" or "No anomalies detected").
    """
    # Character-to-index mapping (same as used during training)
    char_to_idx = {chr(i): i - 96 for i in range(97, 123)}  # 'a' -> 1, ..., 'z' -> 26
    char_to_idx['<pad>'] = 0

    # Convert characters in the sequence to indices
    seq_idx = [char_to_idx.get(c, 0) for c in seq.lower() if c in char_to_idx]

    # Pad the sequence
    padded_seq = pad_sequences([seq_idx], maxlen=max_len, padding='post')

    # Predict using the trained model
    prediction = nlp_model.predict(padded_seq)
    return "Anomaly detected" if prediction[0] > 0.5 else "No anomalies detected."


def extract_characters(image_path):
    """
    Extracts individual characters from the uploaded image while avoiding extra detections.
    Filters contours based on size and aspect ratio to ensure only valid characters are detected.

    Args:
        image_path (str): Path to the uploaded image.
    """
    # Clear the output directory if it exists
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    char_count = 0
    for contour in contours:
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Define size and aspect ratio constraints to filter noise
        if 10 < w < 100 and 10 < h < 100:  # Character size range
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 1.5:  # Aspect ratio range for typical characters
                # Extract and resize the character image
                char_img = binary_img[y:y + h, x:x + w]
                char_img = cv2.resize(char_img, (28, 28))

                # Save the character image
                char_img_name = os.path.join(OUTPUT_DIR, f'char_{char_count}.png')
                Image.fromarray(char_img).save(char_img_name)
                char_count += 1


def generate_pdf_report(result, predictions_array, nlp_output, image_path):
    """
    Generates a formal PDF report with the classification results, NLP output, and the uploaded image.
    """
    pdf_path = PDF_FILE_PATH
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title and Subtitle
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Dyslexia Detection Report")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 80, "Comprehensive Analysis of Handwriting Sample")

    # Date and Time
    c.setFont("Helvetica", 10)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(50, height - 110, f"Generated on: {current_time}")

    # Result Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 140, "Detection Summary:")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 160, f"Overall Assessment: {result}")

    # NLP Output
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 190, "NLP Model Analysis:")
    c.setFont("Helvetica", 12)
    y_pos = height - 210
    for line in nlp_output.split("\n"):
        c.drawString(70, y_pos, line)
        y_pos -= 20
        if y_pos < 50:
            c.showPage()
            y_pos = height - 50

    # Character Classification Details
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos - 30, "Character Classification Details:")
    c.setFont("Helvetica", 12)
    y_pos -= 50
    for idx, pred in enumerate(predictions_array, start=1):
        c.drawString(70, y_pos, f"Character {idx}: {pred}")
        y_pos -= 20
        if y_pos < 50:
            c.showPage()
            y_pos = height - 80

    # Add Uploaded Image
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Uploaded Handwriting Sample:")
    try:
        c.drawImage(image_path, 50, height - 400, width=500, height=350, preserveAspectRatio=True)
    except Exception as e:
        c.drawString(50, height - 450, f"Unable to display image: {e}")

    c.save()
    return pdf_path

def delete_files_after_delay(files, delay):
    """
    Deletes specified files after a delay.
    """
    time.sleep(delay)
    for file in files:
        if os.path.exists(file):
            os.remove(file)

# ------------------------------ Routes -------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    uploaded_file.save(IMAGE_PATH)

    try:
        extract_characters(IMAGE_PATH)
        predictions_array = [predict_image(os.path.join(OUTPUT_DIR, f), cnn_model)
                             for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]

        result = "Dyslexia Detected" if predictions_array.count('Reversal') > predictions_array.count('Normal') else "No Dyslexia Detected"
        pdf_path = generate_pdf_report(result, predictions_array, detect_sequence_anomaly("sample_sequence"), IMAGE_PATH)
    finally:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    return jsonify({'result': result, 'pdf_report': pdf_path})

@app.route('/download_report', methods=['GET'])
def download_report():
    if os.path.exists(PDF_FILE_PATH):
        return send_file(PDF_FILE_PATH, as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=501, debug=True, use_reloader=False)