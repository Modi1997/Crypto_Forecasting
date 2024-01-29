import math
import sys

sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/API_and_Data')
from API_and_Data.get_live_data import *

# pycharm settings to display more columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


def append_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the time to datetime format and generates the following columns:
    Date, Year, Month, Day, Week_of_Year

    :param df: cryptocurrency
    :return: dataframe with additional date features
    """

    # getting the date, year, month, day and week of the year of the given data
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week

    # as we have ['Time'] that is a duplicate drop the column ['Date']
    df = df.drop(columns=['Date'], axis=1)

    return df


def create_trigonometric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converting the Year, Month and Day columns to sin and cos to bring their value ranges closer.
    This is a data preparation technique as the Neural Networks learn better to lower and closed values

    :param df: dataframe with Year, Month, Day
    :return: df with normalised values
    """

    # create sine and cosine for the Year, Month and Day columns
    df['Year_sin'] = df['Year'].apply(lambda x: math.sin(2*math.pi*x/2024))
    df['Year_cos'] = df['Year'].apply(lambda x: math.cos(2*math.pi*x/2024))
    df['Month_sin'] = df['Month'].apply(lambda x: math.sin(2*math.pi*x/12))
    df['Month_cos'] = df['Month'].apply(lambda x: math.cos(2*math.pi*x/12))
    df['Day_sin'] = df['Day'].apply(lambda x: math.sin(2*math.pi*x/12))
    df['Day_cos'] = df['Day'].apply(lambda x: math.cos(2*math.pi*x/12))

    return df