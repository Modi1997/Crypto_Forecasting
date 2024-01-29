# This is a python file where we are testing the functionalities of various functions and provide usage examples

import sys
sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.technical_indicators import *


############################## RSI ##############################
rsi = RSI(frame('BTCUSDT', '1h', '24h'))
############################## EMA ##############################
ema = EMA(frame('BTCUSDT', '1h', '34h'))
############################## MACD #############################
macd = MACD(frame('BTCUSDT', '1h', '34h'))
# macd has 3 arrays (buy, sell, total(buy-sell))
macd_buy = macd[-1][-2:]
macd_sell = macd[-2][-2:]
macd_total = macd[-3][-2:]

# print results
print('RSI:', '\n', rsi[-3:], '\n')
print('EMA:', '\n', ema[-3:], '\n')
print('MACD (sell, buy, difference):', '\n', macd_sell, '\n', macd_buy, '\n', macd_total)