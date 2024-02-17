from API_and_Data.binance_api_request import *


def open_position(symbol: str, side: str, qty: int):
    """
    This function makes MARKET live orders to your account based on

    :param symbol: pair symbol such as BTCUSDT
    :param side: BUY or SELL
    :param qty: quantity of symbol we want to trade
    :return: information of the order
    """

    order = client.create_order(symbol=symbol,
                                side=side,
                                type='MARKET',
                                quantity=qty)
    return order