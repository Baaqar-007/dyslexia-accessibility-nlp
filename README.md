# Dyslexia Accessibility NLP Project

## Overview

This project provides an integrated solution for detecting and addressing dyslexia through **Machine Learning (ML)** and **Natural Language Processing (NLP)**. It leverages state-of-the-art techniques to analyze handwriting and textual input for dyslexic patterns, enabling early diagnosis and accessibility improvements.

Key features include handwriting analysis using **Convolutional Neural Networks (CNNs)** and **Multi-Layer Perceptrons (MLPs)**, as well as anomaly detection in written text through advanced NLP models. The system is complemented by a user-friendly web interface for seamless interaction and reporting.

---

## Key Features

### 1. **Handwriting Classification**
- CNN-based models identify handwritten characters and detect patterns such as reversals, omissions, or misalignments indicative of dyslexia.

### 2. **Dyslexia Detection**
- Combines MLP-based character classification with CNN-driven letter pattern recognition to identify dyslexic tendencies.
- Generates detailed analysis reports for educators and professionals.

### 3. **Sequence Anomaly Detection with NLP**
- NLP models trained to recognize textual sequence anomalies, aiding in the analysis of dyslexic writing patterns.
- Includes LSTM-based models (work in progress).

### 4. **Interactive User Interface**
- A **Flask**-powered web application for uploading handwriting samples or text inputs.
- Displays real-time analysis results with downloadable PDF reports.

---

## Applications

- **Educational Tools**: Assistive technology for educators to identify dyslexic patterns in students’ handwriting.
- **Accessibility Enhancements**: Provides tailored recommendations for users with dyslexia, improving learning outcomes.
- **Diagnostic Aid**: Supports early diagnosis for therapists, educators, and healthcare professionals, enabling personalized interventions.

---

## Technologies Used

| **Category**            | **Technology**               |
|--------------------------|------------------------------|
| **Machine Learning**     | TensorFlow (Keras), CNN, MLP |
| **Natural Language Processing** | LSTM-based sequence models |
| **Web Framework**        | Flask                        |
| **PDF Reporting**        | ReportLab                    |
| **Image Processing**     | OpenCV for character extraction |

---

## Future Work

- Expand NLP capabilities for sequence anomaly detection.
- Optimize models for improved accuracy and performance.
- Enhance the web interface for accessibility and usability.

---

## Conclusion

This project merges ML and NLP to deliver a robust tool for dyslexia detection and accessibility improvement. By combining cutting-edge technology with a user-centric approach, it offers a practical solution for educators, therapists, and researchers aiming to enhance learning environments and provide timely diagnoses.
