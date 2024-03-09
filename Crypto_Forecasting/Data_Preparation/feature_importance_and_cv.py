import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from Data_Preparation.final_df import get_df
from Data_Preparation.subsets_and_target import create_target_variable
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# data
df = get_df("BTCUSDT", "4h", "6000h")
# target, features
coin_df, target, features = create_target_variable(df)

# Assuming df is your DataFrame
X = df[features]
y = df[target]

# Create a RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X, y)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()


# Cross-Validation
# Assuming X and y are your features and target variable
X = df.drop('Close', axis=1).values
y = df['Close'].values

# Set up TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define your LSTM model
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(X.shape[1], 1), return_sequences=True))
model.add(LSTM(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Perform cross-validation
cv_scores = cross_val_score(model, X.reshape(X.shape[0], X.shape[1], 1), y, cv=tscv, scoring='neg_mean_squared_error')

# Calculate RMSE from negative MSE
rmse_scores = np.sqrt(-cv_scores)

# Print the cross-validation scores
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean RMSE:", np.mean(rmse_scores))