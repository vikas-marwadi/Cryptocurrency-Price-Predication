import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("Time Series Forecasting Web App")

uploaded_file = st.file_uploader("Upload a CSV file containing a time series dataset", type="csv")

def evaluate_model_streamlit(actual, predicted, model_name):
    mse = mean_squared_error(actual, predicted)
    rmse = sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100

    st.subheader(f"{model_name} Model Evaluation")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MAPE: {mape:.4f}%")

def plot_forecast(original_dates, original_values, forecast_dates, forecast_values, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(original_dates, original_values, label='Original Data', color='blue')
    ax.plot(forecast_dates, forecast_values, label=f'{model_name} Predictions', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(f'Time Series Forecasting with {model_name}')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True, infer_datetime_format=True) # Assuming first column is date/index
    # Convert Unix timestamps (seconds since epoch) to datetime
    df.index = pd.to_datetime(df.index, unit='s')  # 's' for seconds
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if len(df.columns) < 1:
        st.error("The uploaded CSV must contain at least one numerical column for forecasting.")
    else:
        data_column = st.selectbox("Select the column to forecast", df.columns, index=5)
        data = df[data_column].dropna()
        if data.empty:
            st.error(f"The selected column '{data_column}' contains no valid data after dropping NaNs.")
        else:
            data_column = "close"
            st.subheader(f"Time Series Data: {data_column}")
            st.line_chart(data)

            decomposition_type = st.selectbox("Choose Decomposition Type", ["Additive", "Multiplicative"])
            if st.button("Decompose Time Series"):
                try:
                    model = 'additive' if decomposition_type == "Additive" else 'multiplicative'
                    period = st.slider("Set Decomposition Period (e.g., 30 for daily seasonality in monthly data)", 2, len(data) // 5, 30)
                    if period > 1:
                        decomposition = seasonal_decompose(data, model=model, period=period, extrapolate_trend='freq')
                        st.subheader(f"{decomposition_type} Decomposition")
                        fig, axes = plt.subplots(4, 1, figsize=(10, 8))
                        decomposition.observed.plot(ax=axes[0], title='Observed')
                        decomposition.trend.plot(ax=axes[1], title='Trend')
                        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                        decomposition.resid.plot(ax=axes[3], title='Residuals')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Please select a decomposition period greater than 1.")
                except Exception as e:
                    st.error(f"Error during decomposition: {e}")

            st.subheader("Forecasting Models")
            model_choice = st.selectbox("Select a Forecasting Model", ["ARIMA", "Exponential Smoothing (ETS)", "Prophet", "LSTM"])

            train_size = int(len(data) * 0.8)
            train_data, test_data = data[:train_size], data[train_size:]

            if model_choice == "ARIMA":
                p = st.slider("ARIMA (p): Autoregressive order", 0, 5, 2)
                d = st.slider("ARIMA (d): Differencing order", 0, 2, 1)
                q = st.slider("ARIMA (q): Moving Average order", 0, 5, 2)
                if st.button("Train and Predict ARIMA"):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        model_fit = model.fit()
                        predictions = model_fit.forecast(steps=len(test_data))
                        evaluate_model_streamlit(test_data, predictions, "ARIMA")
                        plot_forecast(data.index, data.values, test_data.index, predictions, "ARIMA")
                    except Exception as e:
                        st.error(f"Error training/predicting with ARIMA: {e}")

            elif model_choice == "Exponential Smoothing (ETS)":
                trend_option = st.selectbox("ETS Trend", ["add", "mul", None])
                seasonal_option = st.selectbox("ETS Seasonal", ["add", "mul", None])
                seasonal_periods_ets = st.slider("ETS Seasonal Periods", 2, len(data) // 5, 30) if seasonal_option else None
                if st.button("Train and Predict ETS"):
                    try:
                        model = ExponentialSmoothing(train_data, trend=trend_option, seasonal=seasonal_option, seasonal_periods=seasonal_periods_ets).fit()
                        predictions = model.forecast(len(test_data))
                        evaluate_model_streamlit(test_data, predictions, "ETS")
                        plot_forecast(data.index, data.values, test_data.index, predictions, "ETS")
                    except Exception as e:
                        st.error(f"Error training/predicting with ETS: {e}")

            elif model_choice == "Prophet":
                if st.button("Train and Predict Prophet"):
                    # try:
                    prophet_df_train = pd.DataFrame({'ds': pd.to_datetime(train_data.index), 'y': train_data.values})
                    model = Prophet(daily_seasonality=False)
                    model.fit(prophet_df_train)
                    future = pd.DataFrame({'ds': pd.to_datetime(test_data.index)})
                    forecast = model.predict(future)
                    predictions = forecast['yhat'].values
                    evaluate_model_streamlit(test_data, predictions, "Prophet")
                    plot_forecast(data.index, data.values, test_data.index, predictions, "Prophet")
                    # except Exception as e:
                    #     st.error(f"Error training/predicting with Prophet: {e}")

            elif model_choice == "LSTM":
                seq_length = st.slider("LSTM Sequence Length", 10, 100, 30)
                n_epochs = st.slider("LSTM Epochs", 10, 100, 20)
                if st.button("Train and Predict LSTM"):
                    # try:
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

                    def create_sequences(data, seq_length):
                        X, y = [], []
                        for i in range(len(data) - seq_length):
                            X.append(data[i:i + seq_length])
                            y.append(data[i + seq_length])
                        return np.array(X), np.array(y)

                    X, y = create_sequences(scaled_data, seq_length)
                    train_size_lstm = int(len(X) * 0.8)
                    X_train_lstm, X_test_lstm = X[:train_size_lstm], X[train_size_lstm:]
                    y_train_lstm, y_test_lstm = y[:train_size_lstm], y[train_size_lstm:]

                    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
                    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

                    model_lstm = Sequential()
                    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
                    model_lstm.add(LSTM(50))
                    model_lstm.add(Dense(1))
                    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
                    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=n_epochs, batch_size=32, verbose=0)
                    lstm_pred_scaled = model_lstm.predict(X_test_lstm)
                    lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
                    lstm_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

                    evaluate_model_streamlit(lstm_actual, lstm_pred, "LSTM")

                    # Create a simple integer index for the forecast period
                    forecast_index = np.arange(len(data) - len(lstm_actual), len(data))

                    # plot_forecast(data.index, data.values, forecast_index, lstm_pred, "LSTM")
                    plot_forecast(data.index[-len(lstm_actual):], lstm_actual, data.index[-len(lstm_pred):], lstm_pred, "LSTM")

                    # except Exception as e:
                    #     st.error(f"Error training/predicting with LSTM: {e}")
