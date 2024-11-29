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

### 3. **Sequence Anomaly Detection with NLP (work in progress)**
- NLP models trained to recognize textual sequence anomalies, aiding in the analysis of dyslexic writing patterns.
- Includes LSTM-based models.

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
## Dataset

This project utilizes a variety of datasets designed for handwriting analysis and dyslexia detection:

1. **EMNIST Dataset**  
   - An extension of the MNIST dataset, comprising 814,255 images of handwritten digits and letters across 62 classes.  
   - Widely used for handwriting anomaly detection, particularly in uppercase and lowercase letters.

2. **NIST Special Database 19**  
   - A large dataset of handwritten characters collected from over 500 writers.  
   - Forms the foundation for MNIST and EMNIST, supporting advanced optical character recognition (OCR) tasks.

3. **Dyslexia-Specific Handwriting Samples**  
   - Datasets focused on dyslexic writing patterns, including mirrored letters, reversals, and omissions.  
   - Collected from primary school children and used in studies employing CNN and SVM models for dyslexia detection.

---
## Conclusion

This project merges ML and NLP to deliver a robust tool for dyslexia detection and accessibility improvement. By combining cutting-edge technology with a user-centric approach, it offers a practical solution for educators, therapists, and researchers aiming to enhance learning environments and provide timely diagnoses.

---

## References

- Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). *EMNIST: An Extension of MNIST to Handwritten Letters*. Western Sydney University.  
- Alqahtani, N. D., Alzahrani, B., & Ramzan, M. S. (2023). *Detection of Dyslexia Through Images of Handwriting Using Hybrid AI Approach*. International Journal of Advanced Computer Science and Applications (IJACSA).  
- Isa, I. S., Rahimi, W. N. S., Ramlan, S. A., & Sulaiman, S. N. (2019). *Automated Detection of Dyslexia Symptom Based on Handwriting Image for Primary School Children*. Procedia Computer Science.  


