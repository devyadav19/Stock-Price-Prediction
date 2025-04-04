# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load Stock Data with Technical Indicators
def load_stock_data(ticker, start="2015-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    return df.dropna()

# Preprocess Data for LSTM
def preprocess_data(df, seq_length=50):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']])
    X, Y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        Y.append(scaled_data[i+seq_length])
    X, Y = np.array(X), np.array(Y)
    return X.reshape((X.shape[0], X.shape[1], 1)), Y, scaler

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train Model and Predict
def train_and_predict(ticker, seq_length=50, epochs=50):
    df = load_stock_data(ticker)
    X, Y, scaler = preprocess_data(df, seq_length)
    train_size = int(len(X) * 0.8)
    X_train, Y_train, X_test, Y_test = X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
    model = build_lstm_model((seq_length, 1))
    model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_test, Y_test), verbose=1)
    predicted_prices = scaler.inverse_transform(model.predict(X_test))
    return df, predicted_prices, scaler, model

# Predict Future Prices
def predict_future_prices(model, last_50_days, scaler, days_to_predict=30):
    future_predictions = []
    input_seq = last_50_days.reshape(1, -1, 1)
    for _ in range(days_to_predict):
        next_price = model.predict(input_seq)[0][0]
        future_predictions.append(next_price)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_price]]], axis=1)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Streamlit Web App
def main():
    st.title("📈 Stock Price Prediction App")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()
    if st.button("Predict Stock Prices"):
        st.write(f"Fetching data for {ticker}...")
        df, predicted_prices, scaler, model = train_and_predict(ticker)
        last_50_days = df['Close'].values[-50:].reshape(-1, 1)
        last_50_days_scaled = scaler.transform(last_50_days)
        future_prices = predict_future_prices(model, last_50_days_scaled, scaler, days_to_predict=30)
        st.subheader("📊 Actual vs Predicted Prices")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['Close'].values[-len(predicted_prices):], name="Actual Price", mode='lines'))
        fig.add_trace(go.Scatter(y=predicted_prices.flatten(), name="Predicted Price", mode='lines'))
        st.plotly_chart(fig)
        st.subheader("📈 Future Predictions (Next 30 Days)")
        future_fig = go.Figure()
        future_fig.add_trace(go.Scatter(y=future_prices.flatten(), name="Future Prices", mode='lines', line=dict(dash="dot")))
        st.plotly_chart(future_fig)

# Run the Streamlit App
if __name__ == "__main__":
    main()