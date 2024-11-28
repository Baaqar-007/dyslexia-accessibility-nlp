import numpy as np
from nltk.util import ngrams
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
def load_sequences(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

def preprocess_sequences(data, max_len=5, vocab_size=27):
    char_to_idx = {chr(i): i - 96 for i in range(97, 123)} # Map 'a' -> 1, ...,'z' -> 26
    char_to_idx['<pad>'] = 0

    sequences = []
    labels = []
    for line in data:
        line = line.strip().lower()
        label = 1 if "anomaly" in line else 0 # Add your labeling logic
        seq_idx = [char_to_idx.get(c, 0) for c in line if c in char_to_idx]
        sequences.append(seq_idx)
        labels.append(label)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences, np.array(labels)

def train_sequence_model(data_path, output_model_path):
    data = load_sequences(data_path)
    X, y = preprocess_sequences(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    # Define the sequence model
    model = Sequential([Embedding(input_dim=27, output_dim=16, input_length=X.shape[1]),LSTM(32),Dense(1, activation='sigmoid') ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    # Save the model
    model.save(output_model_path)
if __name__ == "__main__":
    data_path = "sequence_data.txt"
    output_model_path = "nltk_sequence_anomaly_detector.h5"
    train_sequence_model(data_path, output_model_path)
