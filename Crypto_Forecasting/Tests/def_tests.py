# This is a python file where we are testing the functionalities of various functions and provide usage examples

import sys, time
sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.technical_indicators import *
from Data_Preparation.feature_engineering import *
from Data_Preparation.final_df import *


start_time = time.time()

print(get_df('BTCUSDT', '1h', '100h'))

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

######################### Feature engineering ###################
data = get_data('BTCUSDT', '1h', '24h')
# add the date features
df_date = append_date_features(data)
# add the sine and cosine date features
df = create_trigonometric_columns(df_date)
############################## RSI ##############################
rsi = RSI(frame('BTCUSDT', '1h', '24h'))
############################## EMA ##############################
ema = EMA(frame('BTCUSDT', '1h', '34h'))
############################## MACD #############################
macd = MACD(frame('BTCUSDT', '1h', '34h'))
# macd has 3 arrays (buy, sell, total(buy-sell))
macd_total = macd[-1]
macd_sell = macd[-2][-2:]
macd_buy = macd[-3][-2:]

# print results
print('RSI:', '\n', rsi[-3:], '\n')
print('EMA:', '\n', ema[-3:], '\n')
print('MACD (sell, buy, difference):', '\n', macd_buy, '\n', macd_sell, '\n', macd_total)