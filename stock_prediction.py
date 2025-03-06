# full code to save as a .py file to upload in GitHub.


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Fetch Tesla stock data
ticker = 'TSLA'
start_date = '2015-01-01'
end_date = '2024-01-01'

df = yf.download(ticker, start=start_date, end=end_date)

# Initial analysis of Tesla stock price
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()
df['Returns'] = df['Close'].pct_change()

# Data visualization
plt.figure(figsize=(16, 8))
plt.plot(df['Close'], label='Tesla Closing Price')
plt.plot(df['SMA50'], label='50-Day SMA')
plt.plot(df['SMA200'], label='200-Day SMA')
plt.legend()
plt.title('Tesla Stock Price with Moving Averages')
plt.show()

# Preprocessing and sequence creation
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

sequence_length = 60

def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        seq = data[i : i + sequence_length]
        label = data[i + sequence_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(scaled_data, sequence_length)

# Splitting data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define AI models
def build_rnn_model():
    model = Sequential([
        SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        SimpleRNN(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_gru_model():
    model = Sequential([
        GRU(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        GRU(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

# Compile and train models
models = {
    'RNN': build_rnn_model(),
    'LSTM': build_lstm_model(),
    'GRU': build_gru_model()
}

for model_name, model in models.items():
    model.compile(optimizer='adam', loss='mse')
    print(f"Training {model_name} model...")
    start_time = time.time()
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    training_time = time.time() - start_time
    print(f"{model_name} Model Training Time: {training_time:.2f} seconds")
    model.save(f'tesla_{model_name.lower()}_model.h5')

# Evaluate models
predictions = {}
metrics = {}

for model_name, model in models.items():
    model = load_model(f'tesla_{model_name.lower()}_model.h5',compile=False)
    model.compile(optimizer='adam', loss='mse')
    start_infer = time.time()
    pred = model.predict(X_test)
    end_infer = time.time()

    # Inverse transform predictions
    pred = scaler.inverse_transform(pred.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
    mae = mean_absolute_error(y_test_actual, pred)
    r2 = r2_score(y_test_actual, pred)

    # Store results
    predictions[model_name] = pred
    metrics[model_name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Inference Time': end_infer - start_infer
    }

    # Print metrics
    print(f"{model_name} Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Inference Time: {metrics[model_name]['Inference Time']:.2f} seconds\n")

# Plot predictions
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], y_test_actual, label='Actual Price')
for model_name, pred in predictions.items():
    plt.plot(df.index[-len(y_test):], pred, label=f'{model_name} Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Tesla Stock Price Prediction: RNN vs LSTM vs GRU')
plt.show()

# Compare model performance
for model_name, metric in metrics.items():
    print(f"{model_name} Performance:")
    print(f"RMSE: {metric['RMSE']:.2f}")
    print(f"MAE: {metric['MAE']:.2f}")
    print(f"R² Score: {metric['R²']:.2f}")
    print(f"Inference Time: {metric['Inference Time']:.2f} seconds\n")
# Function to load models and generate 5-day predictions
def load_models_and_predict(models_to_load, X_input, scaler, days_ahead=5):
    predictions = {}
    for model_name in models_to_load:
        model = load_model(f'tesla_{model_name.lower()}_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse')
        
        # Ensure X_input is in the correct shape (num_samples, sequence_length, num_features)
        if len(X_input.shape) == 2:
            X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 1))
        
        # Use the last sequence from the input data
        future_input = X_input[-1]  # Shape: (sequence_length, 1)
        future_predictions = []

        for _ in range(days_ahead):
            # Reshape input to match model's expected input shape (batch_size, sequence_length, features)
            pred = model.predict(future_input.reshape(1, sequence_length, 1))
            future_predictions.append(pred[0][0])  # Append the predicted value
            
            # Update input for next prediction by removing the first element and appending the new prediction
            future_input = np.append(future_input[1:], pred, axis=0)

        # Inverse transform predictions to original scale
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        predictions[model_name] = future_predictions

    return predictions

# Load models and generate 5-day predictions
models_to_load = ['RNN', 'LSTM', 'GRU']
predictions = load_models_and_predict(models_to_load, X_test, scaler, days_ahead=5)

# Plot 5-day predictions along with actual values
plt.figure(figsize=(14, 7))

# Get the last 5 actual values from the test set
actual_values = scaler.inverse_transform(y_test[-5:].reshape(-1, 1))

# Generate dates for the last 5 days of the test set
last_date = df.index[-len(y_test)]
dates = pd.date_range(start=last_date, periods=5, freq='B')

# Plot actual values
plt.plot(dates, actual_values, label='Actual Price', marker='o')

# Plot predictions for each model
for model_name, pred in predictions.items():
    plt.plot(dates, pred, label=f'{model_name} Prediction', marker='o')

plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Tesla 5-Day Stock Price Prediction: RNN vs LSTM vs GRU')
plt.grid()
plt.show()
