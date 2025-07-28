import streamlit as st
from modules.data_loader import get_stock_data
from modules.trainer import train_model
from modules.predictor import predict_next_days
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import os

st.title("📊 LSTM Stock Price Predictor")
ticker = st.text_input("Enter Stock Symbol (e.g. AAPL)", value="AAPL")
days = st.slider("Days to Predict", 1, 30, 7)

if st.button("Train & Predict"):
    data = get_stock_data(ticker)
    st.line_chart(data['Close'])

    model, scaler = train_model(data)
    predictions = predict_next_days(data, model, scaler, days=days)

    future_dates = pd.date_range(start=data.index[-1], periods=days+1, freq='B')[1:]
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
    pred_df.set_index('Date', inplace=True)
    
    st.subheader(f"📈 Predicted Prices for Next {days} Days")
    st.line_chart(pred_df)