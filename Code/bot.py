import websocket, json, pprint


SOCKET = 'wss://stream.binance.com:9443/ws/etheur@kline_1m'


opens, highs, lows , closes = [], [], [], []


def on_open(ws):
    print('Opened connection')

def on_close(ws):
    print('Closed connection')

def on_message(ws,message):
    json_message = json.loads(message)
    # pprint.pprint(json_message)

    candle = json_message['k']


    is_candle_close = candle['x']
    close = candle['c']

    if is_candle_close:
        print('candle closed at {}'.format(close))

        opens.append(float(candle['o']))
        closes.append(float(candle['c']))
        highs.append(float(candle['h']))
        lows.append(float(candle['l']))


ws = websocket.WebSocketApp(SOCKET, on_open=on_open,on_close=on_close, on_message=on_message)
ws.run_forever()

