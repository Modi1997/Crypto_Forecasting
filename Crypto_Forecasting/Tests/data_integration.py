import sys
sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Algorithms')
sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.feature_engineering import *

# pycharm settings to display more columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


#####################################################################################################################
####################### (Algorithms.binance_api_request.py) -> (Algorithms.get_live_data.py) ########################
btc_data = get_data("BTCUSDT", "1h", "24h")
#####################################################################################################################


#####################################################################################################################
#################### (Algorithms.get_live_data.py) -> (Data_Preparation.feature_engineering.py) #####################
df = get_data("BTCUSDT", "12h", "10000h")
df_date = append_date_features(df)
btc_df = create_trigonometric_columns(df_date)
print(btc_df)
#####################################################################################################################


#TODO merge technical indicators cols to the final df
#####################################################################################################################
############# (Data_Preparation.feature_engineering.py) -> (Data_Preparation.technical_indicators.py) ###############