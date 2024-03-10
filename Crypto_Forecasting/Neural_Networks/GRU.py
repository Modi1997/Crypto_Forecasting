import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import numpy as np
from Data_Preparation.final_df import get_df
from Data_Preparation.subsets_and_target import create_target_variable
import matplotlib.pyplot as plt
import plotly.express as px


# data
df = get_df("BTCUSDT", "12h", "60000h")
# target, features
coin_df, target, features = create_target_variable(df)
# Extract the 'Close' column for prediction
data = df['Close'].values.reshape(-1, 1)
# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create sequences and labels for training the GRU
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# Define hyperparameters
seq_length = 20
epochs = 50
batch_size = 32
gru_units = 30
dense_units = 1
learning_rate = 0.001

# Create sequences and labels
X, y = create_sequences(data_scaled, seq_length)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(units=gru_units, activation='relu', input_shape=(seq_length, 1)),
    tf.keras.layers.Dense(units=dense_units)
])
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,  # Splitting a portion of training data for validation
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
fig_gru = px.line(df_plot,
              x='Time',
              y=['Actual Close', 'Predicted Close'],
              title='GRU Close Price Prediction',
              labels={'Time': 'Time', 'value': 'Close Price', 'variable': 'Type'})
fig_gru.update_layout(xaxis_title='Time',
                  yaxis_title='Close Price',
                  legend_title='Type',
                  hovermode='x')
fig_gru.show()

# Plot training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize=(10, 6))
plt.plot(range(2, epochs + 1), train_loss[1:], label='Training Loss', marker='o')
plt.plot(range(2, epochs + 1), val_loss[1:], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()