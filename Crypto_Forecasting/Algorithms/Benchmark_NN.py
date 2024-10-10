from Algorithms.LSTM import *
import pandas as pd
import plotly.graph_objects as go

# Rename the columns to distinguish GRU and LSTM predictions
df_gru = df_plot_gru.rename(columns={'Predicted Close': 'GRU Prediction'})
df_lstm = df_plot_lstm.rename(columns={'Predicted Close': 'LSTM Prediction'})

# Merge both DataFrames on 'Time' column
df_combined = pd.merge(df_gru[['Time', 'Actual Close', 'GRU Prediction']],
                       df_lstm[['Time', 'LSTM Prediction']],
                       on='Time', how='inner')

# Create the plot
fig = go.Figure()

# Add traces for Actual Close, GRU Prediction, and LSTM Prediction with custom colors and line widths
fig.add_trace(go.Scatter(x=df_combined['Time'], y=df_combined['Actual Close'],
                         mode='lines', name='Actual Close',
                         line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=df_combined['Time'], y=df_combined['GRU Prediction'],
                         mode='lines', name='GRU Prediction',
                         line=dict(color='red', width=3)))
fig.add_trace(go.Scatter(x=df_combined['Time'], y=df_combined['LSTM Prediction'],
                         mode='lines', name='LSTM Prediction',
                         line=dict(color='green', width=3)))

# Update layout with custom grid color and style
fig.update_layout(
    title='GRU vs LSTM - BTC Close Price Prediction',
    xaxis_title='Time',
    yaxis_title='Close Price',
    legend_title='Type',
    hovermode='x',
    xaxis=dict(showgrid=True, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridcolor='lightgray'),
    plot_bgcolor='white'
)

fig.show()