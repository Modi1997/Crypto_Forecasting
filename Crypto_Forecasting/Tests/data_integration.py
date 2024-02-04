import sys
sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.feature_engineering import *
from Data_Preparation.technical_indicators import *
from Data_Preparation.final_df import *

# pycharm settings to display more columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)


#####################################################################################################################
##################### (API_and_Data.binance_api_request.py) -> (API_and_Data.get_live_data.py) ######################
# btc_data = get_data("BTCUSDT", "1h", "24h")
#####################################################################################################################


#####################################################################################################################
################### (API_and_Data.get_live_data.py) -> (Data_Preparation.feature_engineering.py) ####################
# df = get_data("BTCUSDT", "4h", "6000h")
# df_date = append_date_features(df)
# btc_df = create_trigonometric_columns(df_date)
#print(btc_df)
#####################################################################################################################


#####################################################################################################################
############# (Data_Preparation.feature_engineering.py) -> (Data_Preparation.technical_indicators.py) ###############


# macd = MACD(frame('BTCUSDT', '4h', '6000h'))
#
# btc_df['RSI'] = RSI(frame('BTCUSDT', '4h', '6000h'))
# btc_df['EMA'] = EMA(frame('BTCUSDT', '4h', '6000h'))
# btc_df['MACD'] = macd[-1]
# btc_df['MACD_sell'] = macd[-2]
# btc_df['MACD_buy'] = macd[-3]

df = get_df("BTCUSDT", "4h", "6000h")
print(df)