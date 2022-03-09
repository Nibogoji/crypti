import pandas as pd
import numpy as np
import random
from collections import deque
from trader.plots import TradingGraph


df = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv",index_col=0)
df = df.reset_index()
df.drop(['trades','Market Cap'],axis = 1,inplace=True)
df.rename(columns = {'index':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'},inplace=True)
df = df.sort_values('Date')

granularity = 3 #minuti
year_of_data  = int((365*24*60)/granularity)
lookback_window_size = 50


class TradingEnv:

    def __init__(self, data, initial_balance = 10000, memory = lookback_window_size, Render_range = 100):

        self.data = data
        self.initial_balance = initial_balance
        self.total_timesteps = self.data.shape[0]-1
        self.memory = memory
        self.Render_range = Render_range  # render range in visualization

        self.action_space = np.array([0,1,2])

        self.market_window = deque(maxlen=memory)# window su cui opera il trader di volta in volta
        self.action_history = deque(maxlen=memory) # numero massimo di azioni : 1 per timestep

        self.state_space = (self.memory,
                            10 # o,h,l,c,v,action,action_price,action_quantity
                            )

    def resetEnv(self): # resetta e popola la market history per la window da
        
        self.visualization = TradingGraph(Render_range=self.Render_range)  # init visualization
        self.trades = deque(maxlen=self.memory)  # limited orders memory for visualization

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0

        self.start_step = 0
        self.end_step = self.total_timesteps
        self.current_step = self.start_step

        for i in reversed(range(self.memory)):
            current_step = self.current_step - i
            self.action_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.market_window.append([self.data.loc[current_step, 'Open'],
                                        self.data.loc[current_step, 'High'],
                                        self.data.loc[current_step, 'Low'],
                                        self.data.loc[current_step, 'Close'],
                                        self.data.loc[current_step, 'Volume']
                                        ])

        state = np.concatenate((self.market_window, self.action_history), axis=1)

        return state

    def _next_observation(self):

        self.market_window.append([self.data.loc[self.current_step, 'Open'],
                                    self.data.loc[self.current_step, 'High'],
                                    self.data.loc[self.current_step, 'Low'],
                                    self.data.loc[self.current_step, 'Close'],
                                    self.data.loc[self.current_step, 'Volume']
                                    ])
        obs = np.concatenate((self.market_window, self.action_history), axis=1)
        return obs

    def step(self,action):

        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

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
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': Date, 'High': High, 'Low': Low, 'total': self.crypto_bought, 'type': "buy"})

        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date': Date, 'High': High, 'Low': Low, 'total': self.crypto_sold, 'type': "sell"})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held*current_price

        self.action_history.append([
            self.balance,
            self.net_worth,
            self.crypto_bought,
            self.crypto_sold,
            self.crypto_held
        ])

        reward = self.net_worth-self.prev_net_worth # puntera ad avere guadagni stabili, non a fare meno trades ma piu profittevoli

        if self.net_worth<=self.initial_balance/2:

            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

    def render(self, visualize = True):

        if visualize:
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']

            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)

def Random_game(env, visualize, train_episodes = 50, training_batch_size = 500):

    experiment_net_worth = 0

    for episode in range(train_episodes):

        state = env.resetEnv()

        while True:

            env.render(visualize)
            action = np.random.randint(3,size=1)[0]

            state,reward, done = env.step(action)

            if env.current_step == env.end_step:

                experiment_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", experiment_net_worth / train_episodes)

lookback_window_size = 50
train_df = df[:-720 - lookback_window_size]
test_df = df[-720 - lookback_window_size:]  # 30 days

train_env = TradingEnv(train_df, memory=lookback_window_size)
test_env = TradingEnv(test_df, memory=lookback_window_size)

Random_game(test_env, visualize=True, train_episodes=1, training_batch_size=300)

