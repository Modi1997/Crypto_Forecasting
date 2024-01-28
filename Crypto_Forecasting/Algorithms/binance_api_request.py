# This is a python file where we are installing Binance library
# for client requests and connect our API key with the back-end


# pip install python-binance and import client for API request
from binance.client import Client

# secret key and api connection/pass provided by account on binance
api_key = '*************************************'
api_secret = '*************************************'

# client request
client = Client(api_key, api_secret)

# client account all info
client_account = client.get_account()

# client_keys of the dictionary
client_keys = client.get_account().keys()
# get only the cryptocurrency assets
account_info = client_account["balances"]

# print only the assets with over 0 possess
def balance_free() -> object:
    """
    This function once it runs it returns all the cryptocurrency assets with greater than 0 amount

    :return: prints the assets with over 0 balance
    """

    for balance in account_info:
        if balance["free"] not in ('0.0', '0.00', '0.00000000'):
            print(balance)
    return ''