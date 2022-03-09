import pandas as pd
from binance.client import Client
from Credentials import credentials
import datetime

client = Client(api_key=credentials.key, api_secret=credentials.secret)
print('Logged in')


exchange_info = client.get_exchange_info()
account_info = client.get_account()

portfolio = [account_info['balances'][i] for i in range(len(account_info['balances'])) if float(account_info['balances'][i]['free']) > 0]

def get_trades(portfolio):

    traded_symbols = []


    for a in range(len(portfolio)):

        sym = portfolio[a]['asset']

        for asset in range(len(exchange_info['symbols'])):

            if exchange_info['symbols'][asset]['baseAsset']==sym:

                traded_symbols.append(exchange_info['symbols'][asset]['symbol'])

    trades = []
    for t in traded_symbols:

        trades.append(client.get_my_trades(symbol = t))

    trades = [t for t in trades if t != []]

    return trades

trades = get_trades(portfolio)


def calculate_balance(portfolio,riferito_a = 'USDT', fiat = 'EUR'):

    balance = 0

    for a in range(len(portfolio)):
        try:
            try:
                if portfolio[a]['asset'] != riferito_a:
                    asset_balance = float(client.get_asset_balance(asset=portfolio[a]['asset'])['free'])
                    price = float(client.get_avg_price(symbol=portfolio[a]['asset']+riferito_a)['price'])
                    asset_value = asset_balance*price
                else:
                    asset_value = float(client.get_asset_balance(asset=portfolio[a]['asset'])['free'])
            except: # se il token non ha conversione in usdt prova in eth

                asset_balance = float(client.get_asset_balance(asset=portfolio[a]['asset'])['free'])
                eth_value = float(client.get_avg_price(symbol=portfolio[a]['asset'] + 'ETH')['price'])
                price = float(client.get_avg_price(symbol='ETH' + riferito_a)['price'])

                asset_value = asset_balance*eth_value*price
        except:
            try:

                asset_balance = float(client.get_asset_balance(asset=portfolio[a]['asset'])['free'])
                price = 1/float(client.get_avg_price(symbol= riferito_a+portfolio[a]['asset'])['price'])
                asset_value = asset_balance * price

            except:

                try:
                    asset_balance = float(client.get_asset_balance(asset=portfolio[a]['asset'])['free'])
                    usdt_value = float(client.get_avg_price(symbol=portfolio[a]['asset'] + 'USDT')['price'])
                    price = 1/float(client.get_avg_price(symbol=riferito_a+'USDT')['price'])

                    asset_value = asset_balance * usdt_value * price


                except:

                    print('Not taking into account :',portfolio[a])
                    continue

        if riferito_a == 'BTC':

            balance += asset_value
            euro = float(client.get_avg_price(symbol=riferito_a+fiat)['price'])
            euro_balance = balance*euro

        else:

            balance += asset_value
            euro = 1/float(client.get_avg_price(symbol=fiat+riferito_a)['price'])
            euro_balance = balance*euro

    print('Balance in {} : {}'.format(riferito_a,balance))
    print('Balance in Euro : {}'.format(euro_balance))

    return balance

balance = calculate_balance(portfolio, riferito_a='USDT')