from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from Data_Preparation.final_df import get_df
from Data_Preparation.subsets_and_target import create_target_variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import warnings

# filtering tensorflow and general (alert) warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# data
df = get_df("BTCUSDT", "1d", "1200d")
# target, features
coin_df, target, features = create_target_variable(df)
# Extract the 'Close' column for prediction
data = df['Close'].values.reshape(-1, 1)
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
# horizontal steps
steps = 2


# Function to create sequences and labels for training the LSTM
def create_sequences(data, seq_length):
    """
    Creates sequences of data with a specified length for sequence prediction tasks.

    :param data: The input data sequence
    :param seq_length: The length of each sequence
    :return:
        sequences (numpy.ndarray): Array of input sequences
        labels (numpy.ndarray): Array of corresponding labels
    """

    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# Define hyperparameters
seq_length = 15  # results: 3 bad, 10 normal, 15 good
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
    verbose=1
    )

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Inverse transform the predicted and actual values to the original scale
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the MAE and RMSE
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# Create a DataFrame containing the actual and predicted values along with the corresponding timestamps
df_plot = pd.DataFrame({
    'Time': df.index[-len(y_test):],
    'Actual Close': y_test_original.flatten(),
    'Predicted Close': y_pred_original.flatten()
    })

def calculate_step_size(frequency):
    if frequency.endswith('m'):
        step_size = pd.Timedelta(minutes=int(frequency[:-1]))
    elif frequency.endswith('h'):
        step_size = pd.Timedelta(hours=int(frequency[:-1]))
    elif frequency.endswith('d'):
        step_size = pd.Timedelta(days=int(frequency[:-1]))
    elif frequency.endswith('w'):
        step_size = pd.Timedelta(weeks=int(frequency[:-1]))
    else:
        raise ValueError("Invalid frequency format. Please use 'm' for minutes, 'h' for hours, or 'd' for days.")
    return step_size

# Function to generate next steps index based on the frequency
def generate_next_steps_index(df, num_steps, frequency):
    last_timestamp = df.index[-1]
    step_size = calculate_step_size(frequency)
    next_steps_index = pd.date_range(start=last_timestamp + step_size, periods=num_steps, freq=frequency)
    return next_steps_index

last_sequence = X_test[-1:]
# Predict the next two steps
next_steps = []
for _ in range(2):
    next_step_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
    next_steps.append(next_step_pred)
    last_sequence = np.append(last_sequence[:, 1:, :], next_step_pred.reshape(1, 1, 1), axis=1)

# Inverse transform the predicted values to the original scale
next_steps_original = scaler.inverse_transform(np.array(next_steps).reshape(2, 1))

# Extend the DataFrame to include the forecasted values
next_steps_index = generate_next_steps_index(df, steps, "1d")
next_steps_df = pd.DataFrame({
    'Time': next_steps_index,
    'Predicted Close': next_steps_original.flatten()
})

# Concatenate the forecasted values to the existing DataFrame
df_plot_extended = pd.concat([df_plot, next_steps_df], ignore_index=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df_plot_extended['Time'], y=df_plot_extended[f'Actual Close'], mode='lines',
               name=f'Actual Close'))

# Add predicted close price line
fig.add_trace(
    go.Scatter(x=df_plot_extended['Time'], y=df_plot_extended[f'Predicted Close'], mode='lines',
               name=f'Predicted Close'))

# Add scatter plot for forecasted values
fig.add_trace(
    go.Scatter(x=next_steps_df['Time'], y=next_steps_df[f'Predicted Close'], mode='markers',
               name='Forecasted Values', marker=dict(color='green', size=6)))

# Adjust X-axis range
x_range_max = next_steps_df['Time'].max() + pd.Timedelta(hours=12)
fig.update_xaxes(range=[df_plot_extended['Time'].min(),
                        x_range_max])  # Set X-axis range from the minimum date to the maximum date

fig.update_layout(title=f"<b>Actual and Forecast Price</b>",
                  xaxis_title='Time',
                  yaxis_title='Close Price',
                  legend_title='Type',
                  hovermode='x',)
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