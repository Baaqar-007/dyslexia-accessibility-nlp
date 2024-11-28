# Dyslexia Accessibility NLP Project

## Project Overview

This project aims to provide a comprehensive solution for dyslexia detection and accessibility improvement through the use of machine learning (ML) and natural language processing (NLP) techniques. The system analyzes handwriting samples to detect dyslexic patterns and classifies characters using both Convolutional Neural Networks (CNN) and Multi-Layer Perceptron (MLP) models. The project also incorporates an NLP model for sequence anomaly detection in textual input, further enhancing its diagnostic capabilities.

## Key Features

- **Handwriting Classification:** Utilizes CNN models to classify handwritten characters into categories, identifying dyslexic patterns such as reversals and normal characters.
- **Dyslexia Detection:** By combining MLP-based character classification with CNN for letter pattern recognition, the system identifies dyslexic tendencies and provides detailed reports.
- **NLP Model for Sequence Anomaly Detection:** A sequence model trained to detect anomalies in textual patterns, helping to analyze dyslexic writing further.
- **User Interface:** A Flask-based frontend allows users to upload images, interact with the system, and view the analysis results in real time, including a downloadable PDF report with an analysis of character classification and sequence anomalies.

## Applications

- **Educational Tools:** Assistive technology for identifying dyslexic tendencies in students' handwriting.
- **Accessibility Enhancements:** Provides tailored interventions and accommodations for users with dyslexia, aiding in more effective learning environments.
- **Diagnostic Tool:** Used by educators, therapists, and healthcare professionals to diagnose dyslexia early and provide personalized treatment.

## Technologies Used

- **Machine Learning:** TensorFlow (Keras) for training CNN and MLP models.
- **Natural Language Processing:** Sequence anomaly detection using LSTM-based models.[Needs more work]
- **Interface:** Flask for the web interface.
- **PDF Generation:** ReportLab for generating detailed reports.
- **Image Processing:** OpenCV for character extraction from images.

## Conclusion

This project aims to improve accessibility and provide a practical tool for diagnosing dyslexia. It combines advanced machine learning models with NLP techniques to create an intuitive and functional solution that serves a broad range of educational and diagnostic purposes.
