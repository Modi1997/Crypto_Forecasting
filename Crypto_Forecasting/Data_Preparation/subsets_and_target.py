# In this file we will merge all indicators and feature engineering to create our final df
# We will also split the data to subsets and define our target variable

import time
import sys
from datetime import datetime

sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.final_df import *
from Data_Preparation.technical_indicators import *

# pycharm settings to display more columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


def create_target_variable(df: pd.DataFrame, forecast_lead: int = 1) -> (pd.DataFrame, str):
    """
    This function is designed to create a new target variable for time-series forecasting by shifting the values of
    the original "Close" column by a specified lead time.

    :param df: cryptocurrency df
    :param forecast_lead: number of lead for shifting purpose
    :return: df with the 'Close_lead_1' new col
    """

    # distinct target col and rest of the features
    target_column = "Close"
    features_list = list(df.columns.difference([target_column]))

    # create the 'Close_lead_1' col
    target_name = f"{target_column}_lead_{forecast_lead}"

    # remove the forecast_lead rows as they need to be predicted
    # df[target_name] = df[target_column].shift(-forecast_lead)
    # df = df.iloc[:-forecast_lead]

    return df, target_column, features_list


def split_train_valid_test(data: pd.DataFrame):
    """
    This function is splitting the dataset into 3 subsets: train, valid and test. The proposition between them with the
    given split date variables is 70%/15%/15% respectively.

    :param data: dataframe as an input
    :return: train, valid and testing datasets
    """

    # set dates to loc the dataframe in order to have the 70/15/15 rule
    split_date_1 = datetime(2022, 3, 1)
    split_date_2 = datetime(2023, 2, 10)

    # split the data based on the split date variables from above
    train_data = data.loc[data.index < split_date_1]
    valid_data = data.loc[(split_date_1 <= data.index) & (data.index <= split_date_2)]
    test_data = data.loc[data.index > split_date_2]

    return train_data, valid_data, test_data


# # get btc data with a 4h interval
# btc_data = get_df("BTCUSDT", "4h", "60000h")
# # get new df and target
# df, target, features = create_target_variable(btc_data)
# # get training, validation and testing data proposition
# train_data, valid_data, test_data = split_train_valid_test(btc_data)
#
# # confirm that proposition is 70/15/15
# print(df)
# print(f"Dataframe length: {len(df)}")
# print("Training set proposition", round((len(train_data) / len(df)), 2), "%")
# print("Validation set proposition", round((len(valid_data) / len(df)), 2), "%")
# print("Testing set proposition", round((len(test_data) / len(df)), 2), "%")