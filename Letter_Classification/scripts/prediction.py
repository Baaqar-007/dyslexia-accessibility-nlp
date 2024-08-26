import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib
import os
from load_data import load_and_preprocess_data

def make_predictions_and_save_images(X_val, X_test, Y_val, Y_test, output_path):
    # Load the best model
    mlp = joblib.load(os.path.join(output_path, "mlp_best_model.pkl"))

    # Evaluating on the validation set
    val_predictions = mlp.predict(X_val)
    val_accuracy = accuracy_score(Y_val, val_predictions)
    print(f"Validation Set Accuracy: {val_accuracy:.4f}")

    # Evaluating on the test set
    test_predictions = mlp.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_predictions)
    print(f"Test Set Accuracy: {test_accuracy:.4f}")

    # Creating a directory to save images
    os.makedirs(os.path.join(output_path, "predictions"), exist_ok=True)

    # Saving images with labels and predictions
    for i in range(5):  # Reduced to 5 images to lighten the load
        current_image = X_test[i].reshape((28, 28)) * 255
        prediction = test_predictions[i]
        label = Y_test[i]

        plt.figure()
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.title(f'Prediction: {chr(prediction + 64)}, Label: {chr(label + 64)}')
        plt.savefig(os.path.join(output_path, f"predictions/image_{i}_pred_{chr(prediction + 64)}_label_{chr(label + 64)}.png"))
        plt.close()

if __name__ == "__main__":
    # Set file paths
    data_path = '../dataset/emnist-letters-train.csv'
    output_path = "../output/parameters"
    
    # Load data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_preprocess_data(data_path)
    
    # Make predictions and save images
    make_predictions_and_save_images(X_val, X_test, Y_val, Y_test, output_path)
