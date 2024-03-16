from Data_Preparation.final_df import *


# get data
df = get_data('BTCUSDT', '4h', '60000h')
# get only usefyl cols for faster computation
data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
# get only timeframe needed
data = df['2017-01-01':'2018-12-31']


# Create a DataFrame
df = pd.DataFrame(data)

# Calculate EMA
def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Define parameters
short_window = 5
long_window = 20

# Calculate short and long EMA
df['Short_EMA'] = calculate_ema(df['Close'], short_window)
df['Long_EMA'] = calculate_ema(df['Close'], long_window)

# Generate signals
df['Signal'] = 0
df['Signal'][short_window:] = np.where(df['Short_EMA'][short_window:] > df['Long_EMA'][short_window:], 1, 0)
df['Position'] = df['Signal'].diff()

# Initial investment
initial_investment = 1000

# Simulate buying and selling based on signals
cash = initial_investment
crypto = 0

for i in range(len(df)):
    if df['Position'][i] == 1:  # Buy signal
        crypto += cash / df['Close'][i]
        cash = 0
    elif df['Position'][i] == -1 and crypto > 0:  # Sell signal
        cash += crypto * df['Close'][i]
        crypto = 0

# Algorithmic Trading change
final_value = cash + crypto * df['Close'].iloc[-1]
# Algorithmic Trading ROI
ROI = (final_value) / initial_investment * 100

# Actual change
return_without_strategy = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * initial_investment
# Actual ROI
ROI_without = (return_without_strategy) / initial_investment * 100

print(data['Close'].head(1))
print(data['Close'].tail(1), '\n')

print("Final portfolio value with my strategy: $", round(final_value, 2))
print("Return on investment with my strategy: ", round(ROI, 2), "%", '\n')

print("Final portfolio value without strategy: $", round(return_without_strategy, 2))
print("Return without strategy without strategy: ", round(ROI_without, 2), "%")