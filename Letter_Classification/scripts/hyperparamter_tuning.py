from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
from load_data import load_and_preprocess_data

def perform_hyperparameter_tuning(X_train, Y_train, output_path):
    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(250,), (100, 100)],
        'max_iter': [1200, 1600],
        'alpha': [0.001, 0.01]
    }

    # Initialize the classifier
    mlp = MLPClassifier(random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # Save the best model
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_path, "mlp_best_model.pkl"))
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

if __name__ == "__main__":
    # Set file paths
    data_path = '../dataset/emnist-letters-train.csv'
    output_path = "../output/models"
    
    # Load data
    X_train, _, _, Y_train, _, _ = load_and_preprocess_data(data_path)
    
    # Perform hyperparameter tuning
    perform_hyperparameter_tuning(X_train, Y_train, output_path)
