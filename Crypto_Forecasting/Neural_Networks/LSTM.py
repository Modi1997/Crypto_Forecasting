from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from Data_Preparation.final_df import get_df
from Data_Preparation.subsets_and_target import create_target_variable
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import plotly.express as px
import pandas as pd
import os
import warnings

# filtering tensorflow and general (alert) warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# data
df = get_df("BTCUSDT", "4h", "60000h")
# target, features
coin_df, target, features = create_target_variable(df)

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
seq_length = 20  # results: 3 bad, 10 normal, 15 good
epochs = 50
batch_size = 32
units = 30  # default = 50 but if 30 with 20 sequence there is a good training/validation

# Create sequences and labels
X, y = create_sequences(data_scaled, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=units, activation='relu', input_shape=(seq_length, 1)),
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

# Calculate the MAE and RMSE
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Create a DataFrame containing the actual and predicted values along with the corresponding timestamps
df_plot = pd.DataFrame({
    'Time': df.index[-len(y_test):],
    'Actual Close': y_test_original.flatten(),
    'Predicted Close': y_pred_original.flatten()
})

# Create a line plot using Plotly Express
fig = px.line(df_plot, x='Time', y=['Actual Close', 'Predicted Close'], title='LSTM Close Price Prediction',
              labels={'Time': 'Time', 'value': 'Close Price', 'variable': 'Type'})
fig.update_layout(xaxis_title='Time', yaxis_title='Close Price', legend_title='Type', hovermode='x')

# Show the plot
fig.show()

# Extract loss values from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(2, epochs + 1), train_loss[1:], label='Training Loss', marker='o')
plt.plot(range(2, epochs + 1), val_loss[1:], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()