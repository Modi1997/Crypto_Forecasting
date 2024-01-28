from binance_api_account_request import *
import pandas as pd


def get_data(symbol, interval, lookback):
    """
    This function takes a symbol, an interval, and lookback and returns
    a dataframe with the ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    for the given symbol within the lookback timeframe
    (rows of data = interval / lookback)

    :param symbol: pair symbol such as BTCUSDT
    :param interval: bar or time interval (seconds, minutes or hours)
    :param lookback: seconds, minutes or hours of data to look back
    :return: dataframe
    """

    # creating a df with the client.get_historical_klines() binance function
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback+' min ago GMT'))
    # get only the 6 first columns from the client.get_historical_klines()
    frame = frame.iloc[:,:6]
    # name these columns as their headers are just numbers
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']

    # set index to time (ms) and convert type to float
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)

    return frame

# usage example
# last row of the data is the live price
btc_data = get_data("BTCUSDT", "1h", "24h")