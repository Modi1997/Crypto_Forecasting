import sys
sys.path.append('C:/Users/modio/Crypto_Forecasting/Crypto_Forecasting/Data_Preparation')

from Data_Preparation.feature_engineering import *
from Data_Preparation.technical_indicators import *


# getting data
btc_data = get_data("BTCUSDT", "4h", "60000h")
# add the date features
df_date = append_date_features(btc_data)
# add the sine and cosine date features
btc_df = create_trigonometric_columns(df_date)

# add the RSI indicator to the dataframe
btc_df['RSI'] = RSI(frame('BTCUSDT', '4h', '60000h'))
# add the RSI indicator to the dataframe
btc_df['EMA'] = EMA(frame('BTCUSDT', '4h', '60000h'))
# add the MACD indicator to the dataframe
macd = MACD(frame('BTCUSDT', '4h', '60000h'))
# add the MACD difference (or total) array
btc_df['MACD'] = macd[-1]

# check for null values (indicators should have a few)
old_missing_values = btc_df.isnull().sum()
# columns to replace NaN values in
columns_to_fill = ['RSI', 'EMA', 'MACD']

# replace NaN values with mean for each specified column
for column in columns_to_fill:
    mean_value = btc_df[column].mean()
    btc_df[column] = btc_df[column].fillna(mean_value)

# our final df is ready: btc_df