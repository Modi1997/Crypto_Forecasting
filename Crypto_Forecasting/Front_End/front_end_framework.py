import json

import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from datetime import datetime
import time

from flask import Flask, render_template, request
from Data_Preparation.final_df import get_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from binance.client import Client
from flask import jsonify
from Trading_Strats.Algorithmic_Trading import trading_strategy
from API_and_Data.get_live_data import get_data
from Data_Preparation.technical_indicators import *

app = Flask(__name__)

# api_key = 'siP2VBOq44rbgvHfnfWomRb4dcDY7QbVNwAxauetYXGsG9rqCg7YODo3Cn5I57KS'
# api_secret = 'fwgN7NuEXn8hgpBkjVsGs8sYCyqcWRWFv1OkC7jqAepQLLJ5Tehs3vKmifHD7jaS'


trading_history = []
def trading_strategy(symbol, qty: int, entried=False):
    """
    This is an EMA algorithmic trading strategy where we are following the trend (downtrend/uptrend)

    :param symbol: pair symbol such as BTCUSDT
    :param qty: quantity we wish to trade
    :param entried: determines whether you have a LONG position or not
    :return: information of the BUY or SELL trade
    """

    if not entried:
        df = get_data(symbol, '1h', '60h')
        ema = EMA(frame(symbol, '1h', '600h'))

        if (ema.iloc[-1] <= df['Close'].iloc[-1]).all():
            order = client.create_order(symbol=symbol,
                                        side='BUY',
                                        type='MARKET',
                                        quantity=qty)
            now = datetime.now()
            dt_string = now.strftime("%d/%m %H:%M:%S")
            order_info = {
                'Symbol': order['symbol'],
                'Position': order['side'],
                'Quantity': order['origQty'],
                'Price': order['fills'][0]['price']
            }
            message = f"{dt_string} {order_info}"
            trading_history.append(message)
            entried = True

    if entried:
        while True:
            df = get_data(symbol, '1h', '60h')
            ema = EMA(frame(symbol, '1h', '600h'))

            if (ema.iloc[-1] > df['Close'].iloc[-1]).all():
                order = client.create_order(symbol=symbol,
                                            side='SELL',
                                            type='MARKET',
                                            quantity=qty)
                now = datetime.now()
                dt_string = now.strftime("%d/%m %H:%M:%S")
                order_info = {
                    'Symbol': order['symbol'],
                    'Position': order['side'],
                    'Quantity': order['origQty'],
                    'Price': order['fills'][0]['price']
                }
                message = f"{dt_string} {order_info}"
                trading_history.append(message)
                break


@app.route('/login', methods=['POST'])
def login():
        api_key = request.form['api_key']
        api_secret = request.form['api_secret']
        client = Client(api_key, api_secret)
        client_account = client.get_account()
        account_info = client_account["balances"]
        balances = []
        for balance in account_info:
            if float(balance["free"]) > 0:
                balances.append(f"{balance['asset']} : {balance['free']}")
        print(balances)
        return jsonify(balances)


@app.route('/start_algorithmic_trading', methods=['POST'])
def start_algorithmic_trading():
    global trading_history
    symbol = request.form['symbol']
    qty = int(request.form['quantity'])

    started = True
    while started:
        now = datetime.now()
        dt_string = now.strftime("%d/%m %H:%M:%S")
        trading_history = [f"Algorithmic Trading Started at: {dt_string}"]
        started = False

    while True:
        trading_strategy(symbol, qty)
        time.sleep(60)


def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# Function to generate the plotly chart
def calculate_signals(y_pred_original, y_test_original, macd_value, ema_value):
    # Signal 1: AI Forecasting Evaluation
    if y_pred_original[-1] > y_test_original[-1]:
        signal1_text = 'BUY'
        signal1_color = 'green'
    else:
        signal1_text = 'SELL'
        signal1_color = 'red'

    # Signal 2: Technical Analysis Signal
    if macd_value > 0 and ema_value < y_test_original[-1]:
        signal2_text = 'BUY'
        signal2_color = 'green'
    elif macd_value < 0 and ema_value > y_test_original[-1]:
        signal2_text = 'SELL'
        signal2_color = 'red'
    else:
        signal2_text = 'NEUTRAL'
        signal2_color = 'grey'

    return (signal1_text, signal1_color), (signal2_text, signal2_color)


def generate_plot(y_test_original, y_pred_original, df_index, currency_pair):
    df_plot = pd.DataFrame({
        'Time': df_index,
        f'Actual Close of {currency_pair}': y_test_original.flatten(),
        f'Predicted Close of {currency_pair}': y_pred_original.flatten()
    })
    fig = px.line(df_plot, x='Time', y=[f'Actual Close of {currency_pair}', f'Predicted Close of {currency_pair}'],
                  title=f"<b>Actual and Forecast Price of {currency_pair}</b>",
                  labels={'Time': 'Time', 'value': 'Close Price', 'variable': 'Type'})
    fig.update_layout(xaxis_title='Time', yaxis_title='Close Price', legend_title='Type', hovermode='x')

    # Update title position
    fig.update_layout(title_x=0.5)
    # Increase the height of the chart
    fig.update_layout(height=540)
    return fig


@app.route('/', methods=['GET', 'POST'])
def index():
    global trading_history
    if request.method == 'POST':
        cryptocurrency_pair = request.form['cryptocurrency_pair']
        interval = request.form['interval']
        timeframe = request.form['timeframe']
        df = get_df(cryptocurrency_pair, interval, timeframe)
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        X, y = create_sequences(data_scaled, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=30, activation='relu', input_shape=(20, 1)),
            tf.keras.layers.Dense(units=1)])
        model.compile(optimizer='adam', loss='mean_squared_error')
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1)
        y_pred = model.predict(X_test)
        y_pred_original = scaler.inverse_transform(y_pred)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        fig = generate_plot(y_test_original, y_pred_original, df.index[-len(y_test):], cryptocurrency_pair)

        macd_value = df['MACD'][-1]
        rsi_value = df['RSI'][-1]
        ema_value = df['EMA'][-1]

        signal1, signal2 = calculate_signals(y_pred_original, y_test_original, macd_value, ema_value)
        print(trading_history)
        return render_template('index.html', plot=fig.to_html(), macd_value=macd_value, rsi_value=rsi_value,
                               ema_value=ema_value, signal1=signal1, signal2=signal2, y_test_original=y_test_original,
                               y_pred_original=y_pred_original)

    return render_template('index.html', plot=None)


@app.route('/get_trading_history', methods=['GET'])
def get_trading_history():
    global trading_history
    return json.dumps(trading_history)


if __name__ == '__main__':
    app.run(debug=True)