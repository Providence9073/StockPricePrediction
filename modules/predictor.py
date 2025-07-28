import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def predict_next_days(data, model, scaler, window=60, days=7):
    scaled = scaler.transform(data)
    inputs = scaled[-window:].reshape(1, window, 1)
    predictions = []
    
    for _ in range(days):
        pred = model.predict(inputs)[0][0]
        predictions.append(pred)
        inputs = np.append(inputs[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))