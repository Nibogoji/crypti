import pandas as pd
import numpy as np
import random
from collections import deque
from trader.plots import TradingGraph
import ta

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=10000, lookback_window_size=50, Render_range=100):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range  # render range in visualization


        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size=0):
        self.visualization = TradingGraph(Render_range=self.Render_range)  # init visualization
        self.trades = deque(maxlen=self.Render_range)  # limited orders memory for visualization

        self.last_rsi = [0]
        self.last_stoch_k = [0]

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action, tax = 0):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close'])
        Date = self.df.loc[self.current_step, 'Date']  # for visualization
        High = self.df.loc[self.current_step, 'High']  # for visualization
        Low = self.df.loc[self.current_step, 'Low']  # for visualization

        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance / 100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price + tax*self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': Date, 'High': High, 'Low': Low, 'total': self.crypto_bought, 'type': "buy"})

        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price - tax*self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date': Date, 'High': High, 'Low': Low, 'total': self.crypto_sold, 'type': "sell"})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        # Write_to_file(Date, self.orders_history[-1])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self._next_observation()
        print(self.current_step)
        print('P&L : ',self.net_worth)

        return obs, reward, done

    # render environment
    def render(self, visualize=False):
        # print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']

            Last_rsi = self.last_rsi[-1]
            Last_stoch_k = self.last_stoch_k[-1]

            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades, Last_rsi,Last_stoch_k)

    def strategy(self, min_steps = 12):
        if self.current_step> self.start_step + min_steps:

            kama = ta.momentum.kama(self.df.loc[:self.current_step,'Close'], pow1=4, pow2=12)
            # TA's RSI
            rsi = ta.momentum.rsi(kama)
            # TA's Stochastic Oscillator
            stoch_k = ta.momentum.stoch(self.df.loc[:self.current_step,'High'],
                                        self.df.loc[:self.current_step,'Low'],
                                        self.df.loc[:self.current_step,'Close'])
            self.last_rsi.append(rsi.iloc[-1])
            self.last_stoch_k.append(stoch_k.iloc[-1])

            if rsi.iloc[-1] <= 10 and stoch_k.iloc[-1] < 10:
               action = 1

            elif rsi.iloc[-1] > 85 and stoch_k.iloc[-1] > 85:
               action = 2
            else:
                action = 0
        else:
            action = 0

        return action







def Random_games(env, visualize, train_episodes=1, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)
        while True:
            env.render(visualize)

            action = env.strategy()

            state, reward, done = env.step(action, tax = 0.001)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth / train_episodes)

df = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
df = df.reset_index()
df.drop(['trades','Market Cap'],axis = 1,inplace=True)
df.rename(columns = {'index':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
df = df.sort_values('Date')

steps_in_day = 480
lookback_window_size = 50
train_df = df[:-steps_in_day*7 - lookback_window_size]
test_df = df[-steps_in_day*7 - lookback_window_size:]  # 30 days

train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(test_env, visualize=True, train_episodes=1, training_batch_size=300)


