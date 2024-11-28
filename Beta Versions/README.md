# Beta Versions: Transition to Final Version

## Beta Version 1: Early Stage Development

The first iteration of the project focused on initial exploration and feature development. The core functionality was built around using a simple Multi-Layer Perceptron (MLP) to classify individual letters in a handwritten text. The aim was to explore basic character classification and provide initial feedback for early testing.

### Key Features in Beta Version 1:
- **Character Classification with MLP:** A basic MLP model was trained to classify characters.
- **Initial Image Preprocessing:** Simple techniques were applied to extract characters from uploaded images.
- **Basic Reporting:** Users could upload images, and the system would return basic classification results.

## Beta Version 2: Improving Dyslexic Detection

In the second beta version, we transitioned to using a Convolutional Neural Network (CNN) to improve the accuracy of letter recognition. The goal was to enhance the ability to distinguish between "normal" and "reversal" patterns that are indicative of dyslexia.

### Key Features in Beta Version 2:
- **CNN-Based Classification:** Introduced CNN for better feature extraction and classification accuracy.
- **Dyslexic Detection Enhancement:** The system could now classify characters into "Normal" or "Reversal" categories based on CNN output.
- **Image Resizing and Preprocessing Refinements:** Further refinement in how images were processed to ensure better character segmentation.

## Beta Version 3: Integrating NLP for Anomaly Detection

The third beta version integrated an NLP model to detect sequence anomalies. This step was crucial in providing a deeper analysis of handwriting and allowing for the identification of text anomalies typical of dyslexic writing patterns.

### Key Features in Beta Version 3:
- **NLP Anomaly Detection:** The system began using an LSTM-based NLP model to detect anomalies in sequences of characters.
- **PDF Report Generation:** A detailed report summarizing dyslexia detection, classification results, and NLP analysis was added.
- **Flask Web Interface:** A basic web interface was created for easy interaction with the system.

## Final Version: Comprehensive Dyslexia Detection System

The final version integrated all the improvements from the previous beta versions into a comprehensive solution. It combines character classification using CNN, dyslexia detection, and NLP anomaly analysis, providing users with a full diagnostic tool for dyslexia. The final version also includes a polished Flask interface, enhanced error handling, and a clean, user-friendly report generation system.

### Key Features in the Final Version:
- **All-in-One System:** Integrated MLP, CNN, and NLP models for a comprehensive approach to dyslexia detection.
- **User-Friendly Web Interface:** A clean and intuitive Flask web interface for ease of use.
- **Professional PDF Reports:** Automatically generates a formal report that includes detailed analysis of the handwriting, classification results, and sequence anomaly detection.
