import numpy as np
from sklearn.preprocessing import MinMaxScaler
from modules.model_builder import build_model
import os

def prepare_data(data, window=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

def train_model(data):
    X, y, scaler = prepare_data(data)
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    os.makedirs("model", exist_ok=True)
    model.save("model/saved_model.h5")
    return model, scaler