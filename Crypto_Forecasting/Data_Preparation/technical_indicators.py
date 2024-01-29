import talib  # technical analysis library
import pandas as pd
import sys

sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Algorithms')
from Algorithms.binance_api_request import *


def frame(symbol, interval, lookback):
    """
    This function takes a symbol, an interval, and lookback,
    creates a frame (df) and returns the close value that can be used

    :param symbol: pair symbol such as BTCUSDT
    :param interval: bar or time interval (seconds, minutes or hours)
    :param lookback: seconds, minutes or hours of data to look back
    :return: close
    """

    # creating a df with the client.get_historical_klines() binance function
    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback + ' min ago GMT'))
    # get only the 6 first columns from the client.get_historical_klines()
    frame = frame.iloc[:, :5]
    # name these columns as their headers are just numbers
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close']

    # set index to time (ms) and convert type to float
    frame = frame.set_index('Time')
    frame.index = pd.to_datetime(frame.index, unit='ms')
    frame = frame.astype(float)
    # return only the close (plus the index)
    close = frame['Close']

    return close


def RSI(close: object) -> pd.DataFrame:
    """
    This function is creating a technical indicator called RSI (Related Strength Index) and indicates whether the
    asset is over bought or over sold. The 14 first values are NaN as it needs at least 14 values to read. The last 3
    indexes are the most important in order to determine whether there is a turn around or continuation of the trend.

    :param close: input close column
    :return: dataframe with datetime and RSI index (value range: 0,100)
    """

    rsi = talib.RSI(close)
    return rsi


def EMA(close: object) -> pd.DataFrame:
    """
    This function is creating a technical indicator called EMA (Exponential Moving Average) and indicates whether the
    asset is on an uptrend or downtrend. The 30 first values are NaN as it needs at least 14 values to read. As long
    as the actual value is above the EMA then we have an uptrend and respectively below, then a downtrend.

    :param close: input close column
    :return: dataframe with datetime and EMA value
    """

    ema = talib.EMA(close)
    return ema


def MACD(close: object) -> pd.DataFrame:
    """
    This function is creating a technical indicator called MACD (Moving Average Convergence/Divergence) and indicates
    where the cross-over of the trend happens (turn around). The first 33 values are NaN as this technical indicator
    needs data from at least 33 points. Please note that it provides 3 arrays: 1 for the buy, 1 for the sell and which
    indicates the difference between buy-sell. The cross-over happens when the 2nd value from the MACD_total is negative
    and the latest one becomes positive

    :param close: input close column
    :return: dataframe with datetime and MACD
    """

    macd = talib.MACD(close)
    return macd