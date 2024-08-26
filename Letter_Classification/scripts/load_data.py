import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Setting a random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # Loading and preprocessing the dataset
    data = pd.read_csv(file_path).to_numpy()

    # Splitting the dataset into training, validation, and test sets
    X = data[:30000, 1:]   
    Y = data[:30000, 0]           # Labels

    # Splitting the data (70% train, 15% validation, 15% test)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=seed, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=seed, stratify=Y_temp)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
