from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from Data_Preparation.final_df import get_df
from Data_Preparation.subsets_and_target import create_target_variable
from sklearn.metrics import mean_squared_error
import numpy as np

# data
df = get_df("BTCUSDT", "4h", "60000h")
# target, features
coin_df, target, features = create_target_variable(df)

data = df['Close'].values.reshape(-1, 1)

# Assuming df is your DataFrame
# Extract the 'Close' column for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences and labels for training the LSTM
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Define hyperparameters
seq_length = 10  # Length of the input sequences
epochs = 50
batch_size = 32

# Create sequences and labels
X, y = create_sequences(data_scaled, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with validation data
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    verbose=1  # set to 0 for no training progress output
)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Inverse transform the predicted and actual values to the original scale
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_original, label='Actual Close', marker='o')
plt.plot(df.index[-len(y_test):], y_pred_original, label='Predicted Close', marker='o')
plt.title('LSTM Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()