from datetime import datetime, time
from API_and_Data.get_live_data import get_data
from Data_Preparation.technical_indicators import *


def trading_strategy(symbol: str, qty: int, entried=False):
    """
    This function gets a symbol its quantity and based on the trading strategy creates an order (BUY) and afterward
    closes (SELL) this order. It will also display information such as the market price, market fees you paid,
    datetime and more.

    :param symbol: pair symbol such as BTCUSDT
    :param qty: quantity we wish to trade
    :param entried: determines whether you have a LONG position or not
    :return: information of the BUY or SELL trade
    """

    df = get_data(symbol, '15m', '30m')
    close = (df.Close.pct_change() + 1).cumprod() - 1

    macd = MACD(symbol, '15m', '1100m')
    macd1 = macd[-1][-1]

    # we are looking for the right timing to open a LONG position
    if not entried:

        if ((macd1 > 0.00015) and (close[-1] > 0.002)) or (close[-1] > 0.014):
            order = client.create_order(symbol=symbol,
                                        side='BUY',
                                        type='MARKET',
                                        quantity=qty)
            now = datetime.now()
            dt_string = now.strftime("%d/%m %H:%M:%S")
            print("++++++++++++++++++ BUY TIME:", dt_string, " ++++++++++++++++++ ")
            print("\x1b[42m\"+++++++++++ BUY ++++++++++++\"\x1b[0m")
            print("\x1b[42m\"+++++++++ ModiBot ++++++++++\"\x1b[0m")
            print(order)
            entried = True

        else:
            now = datetime.now()
            dt_string = now.strftime("%d/%m %H:%M:%S")
            print("\x1b[47m\">>>>>>>>>>>>>> NO TRADE <<<<<<<<<<<<<<\"\x1b[5m")
            print(dt_string)
            print('\n')

    # while we have a LONG position we are looking to SELL at the right timing
    if entried:

        while True:
            df = get_data(symbol, '15m', '30m')
            macd = MACD(symbol, '15m', '1100m')
            macd1 = macd[-1][-1]
            rsi = RSI(symbol, '5m', '85m')
            rsi_5 = float(rsi.iloc[-1:])

            if rsi_5 > 0:
                sell = (df.Close.pct_change() + 1).cumprod() - 1

                if ((macd1 < -0.00005) and (sell[-1] < - 0.001)) or (sell[-1] < -0.014):
                    order = client.create_order(symbol=symbol,
                                                side='SELL',
                                                type='MARKET',
                                                quantity=qty)
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m %H:%M:%S")
                    print("--------------------- SELL TIME:", dt_string, " ---------------------")
                    print("\x1b[41m\"----------- SELL ------------\"\x1b[0m")
                    print("\x1b[41m\"---------- MondiBot ----------\"\x1b[0m")
                    print(order, "\n", "\n", "\n")
                    break


# time of Modi_Bot started
now = datetime.now()
dt_string = now.strftime("%d/%m %H:%M:%S")
print("Time Modi_Bot started:", dt_string)

# run the algo to check for orders every X (sleep) seconds
while True:
    trading_strategy('CTSIUSDT', 10)
    time.sleep(30)