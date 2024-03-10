import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from Data_Preparation.final_df import get_df
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# get close data
df = get_df("BTCUSDT", "12h", "60000h")
# get data per day
data = df[df.index.hour == 0]

# Hyperparameter tuning
param_grid = {'p': range(3),
              'd': range(3),
              'q': range(3)}

best_model = None
best_rmse = np.inf
best_params = None

# cross validation
cv = TimeSeriesSplit(n_splits=15)
for train_index, test_index in cv.split(data):
    train, test = data.iloc[train_index], data.iloc[test_index]

# Grid search for the best ARIMA parameters
for params in ParameterGrid(param_grid):
    try:
        model = ARIMA(train['Close'], order=(params['p'], params['d'], params['q'])).fit()
        forecast = model.forecast(steps=len(test))
        rmse = np.sqrt(mean_squared_error(test['Close'], forecast))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params
    except:
        continue

# Print the best parameters
print("Best Parameters:", best_params)

# Forecast using the best model
forecast = best_model.forecast(steps=len(test))

# Evaluate the model
rmse = np.sqrt(mean_squared_error(test['Close'], forecast))
print("RMSE:", rmse)

# actual vs ARIMA prediction close values
fig_arima = px.line(x=test.index,
                    y=[test['Close'], forecast],
                    labels={'x': 'Time', 'y': 'Close Price'},
                    title='ARIMA Forecast vs Actual')
fig_arima.update_traces(name='Actual', selector=dict(name='wide_variable_0'))
fig_arima.update_traces(name='Forecast', selector=dict(name='wide_variable_1'))
fig_arima.update_layout(xaxis_title='Time',
                        yaxis_title='Close Price',
                        legend_title='Type',
                        hovermode='x')
fig_arima.show()