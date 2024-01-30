import sys

sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.feature_engineering import *
from Data_Preparation.technical_indicators import *


def get_df(symbol: str, interval: str, lookback: str) -> pd.DataFrame:
    """
    This is function is merging the technical analysis indicators and feature engineering
    columns with the original df given from the get_data function.

    :param symbol: pair symbol such as BTCUSDT
    :param interval: bar or time interval (seconds, minutes or hours)
    :param lookback: seconds, minutes or hours of data to look back
    :return: dataframe
    """

    # getting data
    data = get_data(symbol, interval, lookback)
    # add the date features
    df_date = append_date_features(data)
    # add the sine and cosine date features
    df = create_trigonometric_columns(df_date)

    # add the RSI indicator to the dataframe
    df['RSI'] = round(RSI(frame(symbol, interval, lookback)), 2)
    # add the RSI indicator to the dataframe
    df['EMA'] = round(EMA(frame(symbol, interval, lookback)), 2)
    # add the MACD indicator to the dataframe
    macd = MACD(frame(symbol, interval, lookback))
    # add the MACD difference (or total) array
    df['MACD'] = round(macd[-1], 2)

    # filling the missing values on the indicators
    columns_to_fill = ['RSI', 'EMA', 'MACD']
    # replace NaN values with mean for each specified column
    for column in columns_to_fill:
        mean_value = round(df[column].mean(), 2)
        df[column] = df[column].fillna(mean_value)

    return df