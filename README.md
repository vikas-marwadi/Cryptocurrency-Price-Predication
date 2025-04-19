# Cryptocurrency Time Series Forecasting Project

## Overview

This project investigates the application of various time series forecasting models to predict the closing prices of Bitcoin. It compares the performance of traditional statistical methods like ARIMA, Exponential Smoothing (ETS), and Facebook Prophet with a more advanced deep learning model, Long Short-Term Memory (LSTM) networks. The project also includes the development of an interactive Streamlit web application to make these forecasting techniques accessible to a wider audience.

## Key Features

* **Data Analysis:** Exploratory Data Analysis (EDA) of Bitcoin price data, including visualization, outlier detection, and stationarity testing.
* **Time Series Decomposition:** Implementation of additive and multiplicative decomposition to understand trend, seasonality, and residuals.
* **Forecasting Models:** Implementation and evaluation of four forecasting models:
    * ARIMA (Autoregressive Integrated Moving Average)
    * ETS (Exponential Smoothing - Additive)
    * Prophet (Facebook's forecasting tool)
    * LSTM (Long Short-Term Memory Network)
* **Model Evaluation:** Quantitative evaluation of model performance using key metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
* **Interactive Web Application:** A user-friendly Streamlit application allowing users to:
    * Upload their own CSV time series data.
    * Choose between additive and multiplicative decomposition.
    * Select a forecasting model to train.
    * Visualize actual vs. predicted prices.
    * View model evaluation metrics.
 
## Streamlit Web Application

    * https://cryptocurrencyforecastproject.streamlit.app/
    
## Findings

The study found that the **LSTM model** generally demonstrated superior accuracy in forecasting Bitcoin prices compared to the traditional statistical models. The Streamlit web application provides a user-friendly interface for applying these techniques to various time series datasets.

## Acknowledgements

* https://www.kaggle.com/code/aswingkumar/times-series-bitcoin-prices-based-historical-tre
