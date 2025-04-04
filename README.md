# Stock-Price-Prediction

## Overview
This project is a **Stock Price Prediction App** built using **Streamlit, TensorFlow, and Yahoo Finance**. The application fetches stock market data, applies technical indicators, trains an **LSTM (Long Short-Term Memory) model**, and predicts stock prices.

## Features
- Fetch real-time stock market data using **Yahoo Finance**.
- Calculate technical indicators such as **SMA (Simple Moving Average), RSI (Relative Strength Index), and MACD (Moving Average Convergence Divergence)**.
- Train an **LSTM model** for stock price prediction.
- Visualize **actual vs predicted prices** using **Plotly**.
- Forecast **future stock prices** for the next 30 days.

##  Technologies Used
- **Python**
- **Streamlit** (for the web application)
- **Yahoo Finance API** (`yfinance` for fetching stock data)
- **TA-Lib** (`ta` for technical indicators)
- **TensorFlow/Keras** (for LSTM model)
- **Scikit-learn** (for data scaling)
- **Plotly & Matplotlib** (for data visualization)

##  How It Works
1. Enter the stock ticker symbol (e.g., `AAPL`, `TSLA`, `MSFT`).
2. Click on the **Predict Stock Prices** button.
3. The app will:
   - Fetch historical stock data.
   - Train the LSTM model.
   - Predict stock prices.
   - Display **actual vs predicted** prices.
   - Forecast the **next 30 days**.
4. View interactive visualizations of stock trends.

##  Future Enhancements
- Add more technical indicators
- Improve model performance
- Extend the prediction period
