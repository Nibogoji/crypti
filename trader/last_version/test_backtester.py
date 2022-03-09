import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trader.last_version.filter_modeling_v2 import *
from collections import deque
from sklearn.preprocessing import StandardScaler
import os
import random
import math
from trader.last_version.plots import TradingGraph
import matplotlib
import pickle

# matplotlib.use('TkAgg')

"""
Backtester with integrated predictions
Mutiple horizons, multiple positions, confidence weighted bet, take p stop l

"""


class CustomEnv:

    def __init__(self, df, aap_data5, aap_data10,
                 initial_balance=100000,
                 lookback_window_size=50,
                 base_bet=1000000,
                 training_size=160,
                 retraining=20,
                 prediction_horizon=10,
                 Render_range=100):

        # Define backtesting space

        aap_data5['returns'] = np.nan

        self.last_ohlc = df
        self.last_data = aap_data_5

        self.latest_ohlc = deque(maxlen= training_size)
        self.latest_data_5 = deque(maxlen= training_size)
        self.latest_data_10 = deque(maxlen=training_size)


        self.df = df
        self.aap_data5 = aap_data5
        self.aap_data10 = aap_data10
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.training_size = training_size
        self.Render_range = Render_range  # render range in visualization
        self.initial_price = self.df['Close'].iloc[0]
        self.retraining_step = retraining

        # Position related
        self.base_bet = base_bet

        self.time_out = 0


        self.current_position_5 = 0
        self.position_open_price_5 = 0

        self.current_position_10 = 0
        self.position_open_price_10 = 0

        self.holding_5 = False
        self.holding_10 = False
        self.holding_period_5 = prediction_horizon[0]
        self.holding_period_10 = prediction_horizon[1]

        self.prediction_horizon_5 = prediction_horizon[0]
        self.prediction_horizon_10 = prediction_horizon[1]

        # model related


        self.retraining_points = training_size

        self.confidence_level_5 = 0
        self.confidence_level_10 = 0
        # Action space from 0 to 4, 0 is hold, 1 is buy, 2 is sell, 3 short sell, 4 close short
        self.action_space = np.array([0, 1, 2, 3, 4])
        self.quantity_space = [(i + 1) for i in range(20)]  # descrete bets space



        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

        self.trades_list = []

    # Reset the state of the environment to an initial state
    def reset(self, parameters):
        self.visualization = TradingGraph(Render_range=self.Render_range)  # init visualization
        self.trades = deque(maxlen=self.Render_range)  # limited orders memory for visualization

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance

        self.prev_net_worth = self.initial_balance
        self.quantity_held_5 = 0
        self.quantity_held_10 = 0
        self.quantity_sold_5 = 0
        self.quantity_sold_10 = 0
        self.quantity_bought_5 = 0
        self.quantity_bought_10 = 0
        self.short_held_5 = 0
        self.short_held_10 = 0
        self.net_position = deque(maxlen=self.Render_range)

        self.pure_investment = self.initial_balance
        self.initially_held = 0

        self.start_step = 0
        self.end_step = self.df_total_steps
        self.current_step = self.start_step

        self.volatility =  parameters['var_trained_parameters']['volatility']

        self.current_xgb_model_5 = parameters['model_trained_parameters']['current_xgb_model_5']
        self.current_xgb_model_10 = parameters['model_trained_parameters']['current_xgb_model_10']

        self.current_mlp_model_5 = parameters['model_trained_parameters']['current_mlp_model_5']
        self.current_mlp_model_10 = parameters['model_trained_parameters']['current_mlp_model_10']

        self.current_scaler = parameters['model_trained_parameters']['current_scaler']
        self.current_training_pca_space = parameters['model_trained_parameters']['current_training_pca_space']
        self.current_pcs_receipt = parameters['model_trained_parameters']['current_pcs_receipt']

        # filter related
        self.mean_shift_train = parameters['filter_trained_parameters']['mean_shift_train']
        self.std_shift_train = parameters['filter_trained_parameters']['std_shift_train']
        self.current_xgb_val_score = parameters['filter_trained_parameters']['current_xgb_val_score']
        self.train_test_distances = parameters['filter_trained_parameters']['train_test_distances']
        self.filter_baseline = parameters['filter_trained_parameters']['filter_baseline']
        self.back_to_normal = parameters['filter_trained_parameters']['back_to_normal']
        self.step_to_normal = parameters['filter_trained_parameters']['step_to_normal']
        self.free_will = parameters['filter_trained_parameters']['free_will']
        self.train_test_distances = parameters['filter_trained_parameters']['train_test_distances']
        self.most_important_indexes = parameters['filter_trained_parameters']['most_important_indexes']

        # voting related

        self.best_of_xgb_prediction_5 = parameters['voting_trained_parameters']['best_of_xgb_prediction_5']

        self.best_of_mlp_prediction_5 = parameters['voting_trained_parameters']['best_of_mlp_prediction_5']

        self.best_of_xgb_prediction_10 = parameters['voting_trained_parameters']['best_of_xgb_prediction_10']
        self.best_of_mlp_prediction_10 = parameters['voting_trained_parameters']['best_of_mlp_prediction_10']

        self.returns = parameters['var_trained_parameters']['returns']
        self.step_var = parameters['var_trained_parameters']['step_var']
        self.var_history = parameters['var_trained_parameters']['var_history']

        self.closing_history = parameters['var_trained_parameters']['past_200_close']
        self.orders_history = parameters['var_trained_parameters']['orders_history']
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = parameters['var_trained_parameters']['market_history']
        self.hystorical_data_5 = parameters['var_trained_parameters']['hystorical_data_5']
        self.hystorical_data_10 = parameters['var_trained_parameters']['hystorical_data_10']



        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i

            self.orders_history.append(
                [self.balance, self.net_worth, self.quantity_bought_5 + self.quantity_bought_10,
                 self.quantity_sold_5 + self.quantity_sold_10, self.quantity_held_5 + self.quantity_held_10])

            self.market_history.append([self.df['Open'].iloc[current_step],
                                        self.df['High'].iloc[current_step],
                                        self.df['Low'].iloc[current_step],
                                        self.df['Close'].iloc[current_step]
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df['Open'].iloc[self.current_step],
                                    self.df['High'].iloc[self.current_step],
                                    self.df['Low'].iloc[self.current_step],
                                    self.df['Close'].iloc[self.current_step]
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    def compute_new_returns(self, df, horizon, type='normal'):

        """

        :param df:
        :param type: normal, synt
        :return:
        """
        if type == 'normal':

            df[['returns']].iloc[-horizon - 1] = (df.Price.iloc[-1] - df.Price.iloc[-horizon - 1]) / df.Price.iloc[
                -horizon - 1]

        elif type == 'synt':

            normal_returns = (np.log(df.Price.iloc[-1]) - df.Price.iloc[-horizon - 1]) / df.Price.iloc[-horizon - 1]

            if normal_returns >= 0.005 or normal_returns <= -0.005:
                df[['returns']].iloc[-horizon - 1] = normal_returns
            else:

                df[['returns']].iloc[-horizon - 1] = 0

        return df

    def update_historical_data(self, data):

        new_data_df = pd.concat([data, self.last_data], axis=0)
        new_index = new_data_df.index
        story_to_deque = deque(data.values.tolist(), maxlen=self.training_size)
        story_to_deque.append(self.last_data.values.tolist())
        new_hystorical_data = pd.DataFrame(story_to_deque, index=new_index, columns=new_data_df.columns)

        return new_hystorical_data

    # Execute one time step within the environment
    def step(self, action_5, plot_flag_5, action_10, plot_flag_10, tax=0.001):
        self.quantity_bought = 0
        self.quantity_sold = 0
        self.current_step += 1

        actions = (action_5, action_10)
        horizons = [5, 10]
        action_flag = (plot_flag_5 + ' 5', plot_flag_10 + ' 10')

        # Set the current price to a random price between open and close
        # current_price = random.uniform(self.df['Open'].iloc[self.current_step], self.df['Close'].iloc[self.current_step])
        current_price = self.df['Open'].iloc[self.current_step]
        Date = self.df.index[self.current_step]  # for visualization
        High = self.df['High'].iloc[self.current_step]  # for visualization
        Low = self.df['Low'].iloc[self.current_step]  # for visualization

        for i, (action, horizon) in enumerate(zip(actions, horizons)):
            if action == 0:  # Hold
                pass

            elif horizon == 5 and action == 1 and self.balance > self.initial_balance / 10:

                mwh = (self.base_bet * self.confidence_level_5) / current_price
                # self.quantity_bought_5 = self.quantity_space[min(range(len(self.quantity_space)),
                #                                                key = lambda i: abs(self.quantity_space[i]-mwh))]
                self.quantity_bought_5 = mwh

                self.balance -= self.quantity_bought_5 * current_price + tax * self.quantity_bought_5 * current_price
                self.quantity_held_5 += self.quantity_bought_5
                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_bought_5,
                                    'type': "buy",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})
                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.quantity_bought_5,
                                         'type': "buy",
                                         'flag': action_flag[i]})

                self.current_position_5 = 1
                self.position_open_price_5 = current_price

            elif horizon == 10 and action == 1 and self.balance > self.initial_balance / 10:

                mwh = (self.base_bet * self.confidence_level_10) / current_price
                # self.quantity_bought_10 = self.quantity_space[min(range(len(self.quantity_space)),
                #                                                key=lambda i: abs(self.quantity_space[i] - mwh))]
                self.quantity_bought_10 = mwh

                self.balance -= self.quantity_bought_10 * current_price + tax * self.quantity_bought_10 * current_price
                self.quantity_held_10 += self.quantity_bought_10
                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_bought_10,
                                    'type': "buy",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.quantity_bought_10,
                                         'type': "buy",
                                         'flag': action_flag[i]})

                self.current_position_10 = 1
                self.position_open_price_10 = current_price

            elif horizon == 5 and action == 2 and self.quantity_held_5 > 0:

                mwh = self.quantity_held_5
                self.quantity_sold_5 = mwh

                self.balance += self.quantity_sold_5 * current_price - tax * self.quantity_sold_5 * current_price
                self.quantity_held_5 -= self.quantity_sold_5
                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_sold_5,
                                    'type': "sell",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.quantity_sold_5,
                                         'type': "sell",
                                         'flag': action_flag[i]})

                self.current_position_5 = 0


            elif horizon == 10 and action == 2 and self.quantity_held_10 > 0:

                mwh = self.quantity_held_10
                self.quantity_sold_10 = mwh
                self.balance += self.quantity_sold_10 * current_price - tax * self.quantity_sold_10 * current_price
                self.quantity_held_10 -= self.quantity_sold_10
                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_sold_10,
                                    'type': "sell",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.quantity_sold_10,
                                         'type': "sell",
                                         'flag': action_flag[i]})

                self.current_position_10 = 0

            elif horizon == 5 and action == 3 and self.balance > self.initial_balance / 10:  # OPENING SHORT DOES NOT MODIFY BALANCE

                self.quantity_sold = 0

                mwh = (self.base_bet * self.confidence_level_5) / current_price
                # self.short_held_5 = self.quantity_space[min(range(len(self.quantity_space)),
                #                                                key=lambda i: abs(self.quantity_space[i] - mwh))]
                self.short_held_5 = mwh

                self.balance -= tax * self.short_held_5 * current_price
                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_sold,
                                    'type': "sell",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.short_held_5,
                                         'type': "sell",
                                         'flag': action_flag[i]})

                self.current_position_5 = 3
                self.position_open_price_5 = current_price


            elif horizon == 10 and action == 3 and self.balance > self.initial_balance / 10:

                self.quantity_sold = 0
                mwh = (self.base_bet * self.confidence_level_10) / current_price
                # self.short_held_10 = self.quantity_space[min(range(len(self.quantity_space)),
                #                                                key=lambda i: abs(self.quantity_space[i] - mwh))]
                self.short_held_10 = mwh

                self.balance -= tax * self.short_held_10 * current_price
                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_sold,
                                    'type': "sell",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.short_held_10,
                                         'type': "sell",
                                         'flag': action_flag[i]})

                self.current_position_10 = 3
                self.position_open_price_10 = current_price


            elif horizon == 5 and action == 4 and self.short_held_5 > 0:

                self.quantity_sold = self.short_held_5
                self.balance += self.position_open_price_5 * self.quantity_sold - self.quantity_sold * current_price - \
                                tax * self.quantity_sold * current_price

                self.short_held_5 = 0

                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_sold,
                                    'type': "sell",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.quantity_sold,
                                         'type': "sell",
                                         'flag': action_flag[i]})
                self.current_position_5 = 0

            elif horizon == 10 and action == 4 and self.short_held_10 > 0:
                self.quantity_sold = self.short_held_10
                self.balance += self.position_open_price_10 * self.quantity_sold - \
                                self.quantity_sold * current_price - \
                                tax * self.quantity_sold * current_price
                self.short_held_10 = 0

                self.trades.append({'Date': Date,
                                    'High': High,
                                    'Low': Low,
                                    'total': self.quantity_sold,
                                    'type': "sell",
                                    'flag': action_flag[i],
                                    'Net position': (
                                                self.quantity_held_5 + self.quantity_held_10 - self.short_held_5 - self.short_held_10)})

                self.trades_list.append({'Date': Date,
                                         'High': High,
                                         'Low': Low,
                                         'total': self.quantity_sold,
                                         'type': "sell",
                                         'flag': action_flag[i]})

                self.current_position_10 = 0

            else:

                if horizon == 5:

                    self.holding_5 = False
                    self.holding_period_5 = self.prediction_horizon_5
                    pass
                elif horizon == 10:

                    self.holding_10 = False
                    self.holding_period_10 = self.prediction_horizon_10
                    pass

        self.net_position.append({'Date': Date,
                                  'Net position': (
                                              self.quantity_held_10 + self.quantity_held_5 - self.short_held_5 - self.short_held_10)})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.quantity_held_10 + self.quantity_held_5) * current_price
        self.base_bet = self.balance * 0.75

        self.pure_investment = self.initial_balance / self.initial_price * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.quantity_bought_5 + self.quantity_bought_10,
             self.quantity_sold_5 + self.quantity_sold_10, self.quantity_held_10 + self.quantity_held_5])
        # Write_to_file(Date, self.orders_history[-1])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance / 10:
            done = True
        else:
            done = False

        obs = self._next_observation()
        print(self.current_step)
        print('P&L : ', (self.net_worth - (self.initial_balance + self.initially_held * self.initial_price))
              / (self.initial_balance + self.initially_held * self.initial_price) * 100, ' %')
        print('Pure investment :', self.pure_investment)
        print('Tempo : ', self.df.index[self.current_step])
        print('Current Balance :', self.balance)

        step_returns = np.log(self.df['Open'].iloc[self.current_step]) - np.log(
            self.df['Open'].iloc[self.current_step - 1])
        self.returns.append(step_returns)
        if len(self.returns) >= 12 * 20:
            vola_log_returns = np.std(self.returns)
            var = 1.645 * vola_log_returns * (
                        self.quantity_held_5 + self.quantity_held_10 + self.short_held_5 + self.short_held_10) * current_price
            self.step_var = var
            self.var_history.append(var)
        else:
            self.step_var = 0
            self.var_history.append(0)


        self.hystorical_data_5 = self.update_historical_data(self.hystorical_data_5)
        self.hystorical_data_10 = self.update_historical_data(self.hystorical_data_10)


        self.hystorical_data_5 = self.compute_new_returns(self.hystorical_data_5, 10)
        self.hystorical_data_10 = self.compute_new_returns(self.hystorical_data_10, 80)



        self.current_step += 1


        return obs, reward, done, self.trades_list, self.net_worth, step_returns, self.net_position, self.step_var



    # render environment
    def render(self, visualize=False):
        # print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            Date = self.df.index[self.current_step]
            Open = self.df['Open'].iloc[self.current_step]
            Close = self.df['Close'].iloc[self.current_step]
            High = self.df['High'].iloc[self.current_step]
            Low = self.df['Low'].iloc[self.current_step]

            net_position = self.net_position
            # Render the environment to the screen
            self.visualization.render(Date, Open, High, Low, Close,
                                      self.net_worth, self.trades, self.pure_investment,
                                      self.train_test_distances[-1][0], self.filter_baseline[-1], net_position,
                                      self.step_var,
                                      self.current_pcs_receipt)

    def retraining(self, data5, data10, n_syn_features=2, limit_training=True, reference_points=10, time_out=False):

        syn_features_target_idx = [i for i in range(n_syn_features)]
        syn_features_target_idx.append(-1)


        training_data5 = self.hystorical_data_5.iloc[:self.training_size]
        training_data10 = self.hystorical_data_10.iloc[:self.training_size]


        if time_out:
            training_data5 = data5.iloc[self.latest_data_5.shape[0] - 20:self.current_step]
            training_data10 = data10.iloc[self.latest_data_10.shape[0] - 20:self.current_step]
            self.time_out = 0
            self.free_will = True

        train_Z5, train_features5, train_target5 = make_categorical_target(training_data5, plot=False)
        train_Z10, train_features10, train_target10 = make_categorical_target(training_data10, plot=False)
        for c in range(train_features5.shape[1]):
            if np.array_equal(train_features5.iloc[:, c].values.reshape(1, -1), train_target5.reshape(1, -1),
                              equal_nan=False):
                print('Sanity check failed : Target in features')
                break
        for c in range(train_features10.shape[1]):
            if np.array_equal(train_features10.iloc[:, c].values.reshape(1, -1), train_target10.reshape(1, -1),
                              equal_nan=False):
                print('Sanity check failed : Target in features')
                break

        train_features = train_features5

        categorical_train_target5 = train_Z5.reshape(-1, 1)
        categorical_train_target10 = train_Z10.reshape(-1, 1)

        scaler = StandardScaler()
        scaled_train_features = scaler.fit_transform(train_features)
        X_train_reduced, var_vals, pc_receipt, pcs_space, res, pcs_importance = custom_pca(scaled_train_features,
                                                                                           n_pcs=n_syn_features)
        # What new pcs are made of
        pc_receipt = pd.DataFrame(pc_receipt, columns=['Pc {}'.format(i) for i in range(X_train_reduced.shape[1])])
        # Most n important features to build pcs during the whole experiment to get more or less 12 most importants
        most_important_features = []
        most_important_indexes = []
        importance = []
        for _ in range(int(8 / pcs_space.shape[0])):
            imp = [i for i in pc_receipt.max(axis=0)]
            importance += list(imp * pcs_importance)
            most_important_indexes += [i for i in pc_receipt.idxmax(axis=0)]
            most_important_features += [train_features.columns[i] for i in pc_receipt.idxmax(axis=0)]
            pc_receipt = pc_receipt.drop(most_important_indexes[-pcs_space.shape[0]:], axis=0)

        importance = np.asarray(importance).reshape(1, -1)
        importance = importance / np.sum(importance)

        self.current_pcs_receipt = pd.DataFrame(data=importance, columns=most_important_features)

        reduced_train_dataset5 = np.concatenate((X_train_reduced, categorical_train_target5), axis=1)
        reduced_train_dataset10 = np.concatenate((X_train_reduced, categorical_train_target10), axis=1)

        xgb_val_score_5, xgb_model_5 = run_xgboost_test(reduced_train_dataset5[:, syn_features_target_idx],
                                                        -reference_points)
        mlp_model_5, mlp_val_f1score_5 = run_mlp_test(reduced_train_dataset5[:, syn_features_target_idx],
                                                      -reference_points, rs=8)

        xgb_val_score_10, xgb_model_10 = run_xgboost_test(reduced_train_dataset10[:, syn_features_target_idx],
                                                          -reference_points)
        mlp_model_10, mlp_val_f1score_10 = run_mlp_test(reduced_train_dataset10[:, syn_features_target_idx],
                                                        -reference_points, rs=8)

        self.mean_shift_train = np.mean(res[-reference_points:])
        self.std_shift_train = np.std(res[-reference_points:])

        self.current_xgb_model_5 = xgb_model_5
        self.current_xgb_model_10 = xgb_model_10

        self.current_xgb_val_score = min(xgb_val_score_5, xgb_val_score_10)

        self.current_mlp_model_5 = mlp_model_5
        self.current_mlp_model_10 = mlp_model_10

        self.current_scaler = scaler
        self.current_training_pca_space = pcs_space

        self.most_important_indexes = most_important_indexes

    def filter(self, scaled_test_features, test_projection, pcs, span=3, toll=0.8):

        if pcs >= 3:
            live_test_res = abs(
                np.sum((scaled_test_features ** 2).transpose(), axis=0)
                - np.sum(test_projection ** 2, axis=0)
            )
        else:
            live_test_res = abs(
                np.sum((scaled_test_features[:, self.most_important_indexes] ** 2).transpose(), axis=0)
                - np.sum(test_projection ** 2, axis=0)
            )

        self.train_test_distances.append(abs(np.asarray(self.mean_shift_train) - np.asarray(live_test_res)))
        w_ttd = np.asarray(self.train_test_distances)

        if len(self.train_test_distances) >= span:

            th_up = np.zeros((len(self.train_test_distances), 1))

            for i in range(len(th_up)):
                if i == 0:
                    th_up[i] = np.nan
                else:

                    a = np.mean(w_ttd[:i])
                    b = np.std(w_ttd[:i]) + 0.0001
                    g = b / (a * 2)
                    h = 2 * (a ** 2) / b
                    R = g * np.random.chisquare(h, (10000, 1))
                    th = np.percentile(R, toll)
                    th_up[i] = th

            normal = 2

            w_ttd_ema = pd.Series(w_ttd.reshape(-1, )).ewm(span=span, min_periods=span).mean()
            w_ttd_ema_corrected = np.zeros(w_ttd.shape)

            # Implement a logic to switch between ema and the chisquare filter in case of failure of pca in order to avoid keeping
            # the threshold high even after normal regime is back on the following window

            w_ttd_ema_corrected[-1] = w_ttd_ema.iloc[-1]

            if not self.back_to_normal:

                w_ttd_ema_corrected[-1] = th_up[-1]
                self.step_to_normal += 1
                if self.step_to_normal == normal:
                    self.back_to_normal = True
                else:
                    self.back_to_normal = False

            else:

                if abs(w_ttd_ema_corrected[-2] - w_ttd[-2]) > th_up[-2]:
                    print('Not normal ', self.current_step - 1)
                    w_ttd_ema_corrected[-1] = th_up[-1]
                    self.back_to_normal = False
                    self.step_to_normal = 0

            self.filter_baseline.append(w_ttd_ema_corrected[-1] * toll + (
                    (self.mean_shift_train - w_ttd_ema_corrected[-1]) / self.std_shift_train) * \
                                        math.log2(np.asarray(self.current_xgb_val_score)))


        else:
            self.filter_baseline.append(0)

        if self.filter_baseline[-1] < w_ttd[-1][0]:
            flag = 'Not Reliable'
        else:
            flag = 'Reliable'

        print('Prediction ', flag)
        return flag

    def prediction_pipeline(self, nsf):

        # test_data = self.aap_data5.drop(['returns'], axis=1)
        test_data = self.last_data

        scaled_test_features = self.current_scaler.transform(test_data.to_numpy().reshape(1, -1))

        test_projection = np.dot(self.current_training_pca_space, scaled_test_features.transpose())
        test_reduced = test_projection.transpose()

        # fiter
        filter_response = self.filter(scaled_test_features, test_projection, nsf, toll=0.4)

        xgb_live_predictions_5 = self.current_xgb_model_5.predict(test_reduced)
        mlp_live_predictions_5 = self.current_mlp_model_5.predict(test_reduced)

        xgb_live_predictions_10 = self.current_xgb_model_10.predict(test_reduced)
        mlp_live_predictions_10 = self.current_mlp_model_10.predict(test_reduced)

        return xgb_live_predictions_5[0], mlp_live_predictions_5[0], \
               xgb_live_predictions_10[0], mlp_live_predictions_10[0], \
               filter_response

    def stop_l_take_p(self, current_position, position_open_price, risk_factor_sell=2.5, risk_factor_buy=2):

        current_price = self.df.Open.iloc[self.current_step]

        volatility = np.std(self.df.Close.iloc[self.current_step - 100:self.current_step])
        self.volatility = volatility
        if current_position == 1 and abs(
                position_open_price - current_price) > risk_factor_sell * volatility and current_price < position_open_price:
            action = 2
            sltp = 'Stop Loss'
        elif current_position == 1 and abs(
                position_open_price - current_price) > risk_factor_sell * volatility and current_price > position_open_price:
            action = 2
            sltp = 'Take Profit'

        elif current_position == 3 and abs(
                position_open_price - current_price) > risk_factor_buy * volatility and current_price > position_open_price:
            action = 4
            sltp = 'Short Stop Loss'

        elif current_position == 3 and abs(
                position_open_price - current_price) > risk_factor_buy * volatility and current_price < position_open_price:
            action = 4
            sltp = 'Short Take Profit'

        else:
            action = 0
            sltp = ''

        return action, sltp

    def strategy(self, position='long'):

        vola_p = 0
        vola_d = 0

        if self.current_step >= 0:

            plot_flag_5 = ''
            plot_flag_10 = ''
            nsf = 0
            latest_worths = np.mean(
                [self.orders_history[i][1] for i in range(len(self.orders_history) - 5, len(self.orders_history))])
            relative_latest_worths = latest_worths - self.orders_history[-5][1]
            # print('=============', relative_latest_worths)
            # print(relative_latest_worths < -0.03*self.orders_history[-5][1])
            # print(self.current_step)
            # print(self.train_test_distances)
            if self.current_step == self.retraining_points or self.time_out > 20 or relative_latest_worths < -0.03 * \
                    self.orders_history[-5][1]:

                vola_d = np.std(self.train_test_distances) / np.mean(self.train_test_distances)
                if vola_d > 3:
                    nsf = 12
                else:
                    nsf = 2
                if self.time_out > 20 and self.current_step != self.retraining_points:
                    self.retraining(self.aap_data5, self.aap_data10, limit_training=True, n_syn_features=nsf,
                                    time_out=True)  # 12 false
                elif self.current_step == self.retraining_points:
                    self.retraining(self.aap_data5, self.aap_data10, limit_training=True, n_syn_features=nsf,
                                    time_out=False)  # 12 false
                    self.retraining_points += self.retraining_step
                if relative_latest_worths < -0.03 * self.orders_history[-5][1]:
                    self.retraining(self.aap_data5, self.aap_data10, limit_training=True, n_syn_features=nsf,
                                    time_out=True)  # 12 false

                vola_p = nsf
                print('vola d', vola_d)
                print('pcs', nsf)
            xgb_live_predictions_5, mlp_live_predictions_5, \
            xgb_live_predictions_10, mlp_live_predictions_10, filter_response = self.prediction_pipeline(nsf)

            if self.free_will:

                if xgb_live_predictions_5 == 1:
                    free_action_5 = 1
                    free_plot_flag_5 = 'Opening Position'

                elif xgb_live_predictions_5 == 0:
                    free_action_5 = 3
                    free_plot_flag_5 = 'Short Opening Position'

                else:
                    free_action_5 = 0

                if xgb_live_predictions_10 == 1:
                    free_action_10 = 1
                    free_plot_flag_10 = 'Opening Position'

                elif xgb_live_predictions_10 == 0:
                    free_action_10 = 3
                    free_plot_flag_10 = 'Short Opening Position'

                else:
                    free_action_10 = 0

            self.best_of_xgb_prediction_5.append(xgb_live_predictions_5)
            self.best_of_xgb_prediction_10.append(xgb_live_predictions_10)
            self.best_of_mlp_prediction_5.append(mlp_live_predictions_5)
            self.best_of_mlp_prediction_10.append(mlp_live_predictions_10)

            if len(self.best_of_xgb_prediction_10) == 10:

                latest_predictions_5 = list(self.best_of_xgb_prediction_5) + list(self.best_of_mlp_prediction_5) \
                                       + list(self.best_of_xgb_prediction_10)[:5] + list(
                    self.best_of_mlp_prediction_10)[:5]
                latest_predictions_10 = list(self.best_of_xgb_prediction_10)[5:] + list(self.best_of_mlp_prediction_10)[
                                                                                   5:]

            else:
                latest_predictions_5 = list(self.best_of_xgb_prediction_5) + list(self.best_of_mlp_prediction_5)
                latest_predictions_10 = list(self.best_of_xgb_prediction_10) + list(self.best_of_mlp_prediction_10)

            acceptance_5 = 16 if self.volatility / self.df['Open'].iloc[self.current_step] < 0.1 else 15
            acceptance_10 = 9 if self.volatility / self.df['Open'].iloc[self.current_step] < 0.07 else 11

            if latest_predictions_5.count(0) >= acceptance_5:
                xgb_live_predictions_5 = 0
                self.confidence_level_5 = latest_predictions_5.count(0) / len(latest_predictions_5)

            elif latest_predictions_5.count(1) >= acceptance_5:
                xgb_live_predictions_5 = 1
                self.confidence_level_5 = latest_predictions_5.count(1) / len(latest_predictions_5)
            else:
                xgb_live_predictions_5 = 99

            if latest_predictions_10.count(0) >= acceptance_10:
                xgb_live_predictions_10 = 0
                self.confidence_level_10 = latest_predictions_10.count(0) / len(latest_predictions_10)

            elif latest_predictions_10.count(1) >= acceptance_10:
                xgb_live_predictions_10 = 1
                self.confidence_level_10 = latest_predictions_10.count(1) / len(latest_predictions_10)
            else:
                xgb_live_predictions_10 = 99

            if position == 'short':
                if xgb_live_predictions_10 == 1 and self.holding_10 == False:
                    action_10 = 3
                    plot_flag_10 = 'Short Opening Position'

                elif xgb_live_predictions_10 == 0 and self.holding_10 == False:
                    action_10 = 1
                    plot_flag_10 = 'Opening Position'

                else:
                    action_10 = 0

                if xgb_live_predictions_5 == 1 and self.holding_5 == False:
                    action_5 = 3
                    plot_flag_5 = 'Short Opening Position'

                elif xgb_live_predictions_5 == 0 and self.holding_5 == False:
                    action_5 = 1
                    plot_flag_5 = 'Opening Position'

                else:
                    action_5 = 0

            elif position == 'long':

                if xgb_live_predictions_5 == 1 and self.holding_5 == False:
                    action_5 = 1
                    plot_flag_5 = 'Opening Position'

                elif xgb_live_predictions_5 == 0 and self.holding_5 == False:
                    action_5 = 3
                    plot_flag_5 = 'Short Opening Position'

                else:
                    action_5 = 0

                if xgb_live_predictions_10 == 1 and self.holding_10 == False:
                    action_10 = 1
                    plot_flag_10 = 'Opening Position'

                elif xgb_live_predictions_10 == 0 and self.holding_10 == False:
                    action_10 = 3
                    plot_flag_10 = 'Short Opening Position'

                else:
                    action_10 = 0

        else:
            action_5 = 0
            action_10 = 0
            plot_flag_5, plot_flag_10 = '', ''
            filter_response = 'Not Reliable'

        # Filter action

        # print('Action 10 ===============', action_10)

        sltp_action_5, sltp_flag_5 = 0, ''
        sltp_action_10, sltp_flag_10 = 0, ''

        if filter_response == 'Not Reliable':
            action_5 = 0
            action_10 = 0

        if self.free_will:

            if action_5 == 0 and action_10 == 0:

                action_5 = free_action_5
                plot_flag_5 = 'free ' + free_plot_flag_5
                action_10 = free_action_10
                plot_flag_10 = 'free ' + free_plot_flag_10

                self.free_will = False

            else:
                self.free_will = False

        if action_5 in [1, 3]:
            self.holding_5 = True

        if action_10 in [1, 3]:
            self.holding_10 = True

        if self.holding_5 == True:
            self.holding_period_5 -= 1
            sltp_action_5, sltp_flag_5 = self.stop_l_take_p(self.current_position_5, self.position_open_price_5,
                                                            risk_factor_sell=1.2, risk_factor_buy=1.2)

        if self.holding_10 == True:
            self.holding_period_10 -= 1
            sltp_action_10, sltp_flag_10 = self.stop_l_take_p(self.current_position_10, self.position_open_price_10,
                                                              risk_factor_sell=1.4, risk_factor_buy=1.4)

        if sltp_action_5 == 4:

            action_5 = sltp_action_5
            plot_flag_5 = sltp_flag_5
            self.holding_5 = False
            self.holding_period_5 = self.prediction_horizon_5

        elif sltp_action_5 == 2:

            action_5 = sltp_action_5
            plot_flag_5 = sltp_flag_5
            self.holding_5 = False
            self.holding_period_5 = self.prediction_horizon_5

        if sltp_action_10 == 4:

            action_10 = sltp_action_10
            plot_flag_10 = sltp_flag_10
            self.holding_10 = False
            self.holding_period_10 = self.prediction_horizon_10

        elif sltp_action_10 == 2:

            action_10 = sltp_action_10
            plot_flag_10 = sltp_flag_10
            self.holding_10 = False
            self.holding_period_10 = self.prediction_horizon_10

        if self.holding_period_5 <= 0:

            self.holding_5 == False
            self.holding_period_5 = self.prediction_horizon_5

            if self.current_position_5 == 1:
                if latest_predictions_5.count(
                        0) >= acceptance_5:  # if predictions are still on buy side prolong holding by 1
                    self.holding_5 = True
                    self.holding_period_5 = 1
                    action_5 = 0
                else:
                    action_5 = 2
                    self.holding_5 = False
                    plot_flag_5 = 'Closed position'

            elif self.current_position_5 == 3:

                if latest_predictions_5.count(
                        1) >= acceptance_5:  # if predictions are still on sell side prolong holding by 1
                    self.holding_5 == True
                    self.holding_period_5 = 1
                    action_5 = 0
                else:
                    action_5 = 4
                    self.holding_5 = False
                    plot_flag_5 = 'Short Closed position'

        if self.holding_period_10 <= 0:

            self.holding_10 == False
            self.holding_period_10 = self.prediction_horizon_10

            if self.current_position_10 == 1:
                if latest_predictions_10.count(0) >= 8:  # if predictions are still on buy side prolong holding by 2
                    self.holding_10 = True
                    self.holding_period_10 = 2
                    action_10 = 0
                else:
                    action_10 = 2
                    self.holding_10 = False
                    plot_flag_10 = 'Closed position'

            elif self.current_position_10 == 3:

                if latest_predictions_10.count(0) >= 8:  # if predictions are still on sell side prolong holding by 2
                    self.holding_10 == True
                    self.holding_period_10 = 2
                    action_10 = 0
                else:
                    action_10 = 4
                    self.holding_10 = False
                    plot_flag_10 = 'Short Closed position'

        if action_5 == 0 and action_10 == 0 and self.holding_5 is False and self.holding_10 is False:
            self.time_out += 1

        # rolling_day =  self.df.index[self.current_step]
        #
        # if rolling_day.month == 12 and 29<=rolling_day.day<=31:
        #     if self.current_position_5 ==1:
        #         action_5 = 2
        #     elif self.current_position_5 == 3:
        #         action_5 = 4
        #     else:
        #         action_5 = 0
        #     if self.current_position_10 ==1:
        #         action_10 = 2
        #     elif self.current_position_10 == 3:
        #         action_10 = 4
        #     else:
        #         action_10 = 0
        print('---------------', self.volatility)

        return action_5, plot_flag_5, action_10, plot_flag_10, self.current_step, vola_d, vola_p


def Play_env(parameters, env, visualize, start_render=0, train_episodes=1, training_batch_size=500):
    average_net_worth = 0
    experiment_worths_evo = []
    experiment_returns_evo = []
    for episode in range(train_episodes):
        worth_evolution = []
        returns_evolution = []
        env.reset(parameters)

        vola_ds = []
        vola_ps = []
        while True:

            action_5, plot_flag_5, action_10, plot_flag_10, step, vola_d, vola_p = env.strategy(position='long')
            if step >= start_render:
                env.render(visualize)
            _, _, _, trades_list, net_worth, returns, net_position, var = env.step(action_5, plot_flag_5, action_10,
                                                                                   plot_flag_10, tax=0.001)
            worth_evolution.append(net_worth)
            returns_evolution.append(returns)
            vola_ds.append(vola_d)
            vola_ps.append(vola_p)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth

                break

        experiment_worths_evo.append(worth_evolution)
        experiment_returns_evo.append(returns_evolution)

    # print("average_net_worth:", average_net_worth / train_episodes)
    return trades_list, experiment_worths_evo, experiment_returns_evo, vola_ds, vola_ps


training_size = 480  # 12*4
lookback_window_size = training_size

ohlc = pd.read_csv("C:\\Users\\stesc\\Desktop\\crypti\\Data/LTC.csv", index_col=0)
ohlc.index = pd.to_datetime(ohlc.index)
ohlc = ohlc[~ohlc.index.duplicated(keep='first')]

ohlc = ohlc.loc[:, ['open', 'high', 'low', 'close']]
ohlc.columns = ['Open', 'High', 'Low', 'Close']

tech_ind = pd.read_csv('Code/dati_provvisori/indicatori.csv', index_col=0)
tech_ind.index = pd.to_datetime(tech_ind.index)
tech_ind = tech_ind[~tech_ind.index.duplicated(keep='first')]
tech_ind.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

data_5 = pd.read_csv('Code/dati_provvisori/df_simultaion_v1.csv', index_col=0)
data_5.index = pd.to_datetime(data_5.index)
data_5 = pd.concat([tech_ind, data_5], axis=1)

data_5['slow_returns'] = (data_5.Price.shift(-80) - data_5.Price) / data_5.Price

data_fast = data_5.drop(['target_shift_10', 'returns', 'target', 'cat_target', 'slow_returns'], axis=1).copy()
data_slow = data_5.drop(['target_shift_10', 'returns', 'target', 'cat_target', 'synt_returns'], axis=1).copy()

data_slow.rename(columns={'slow_returns': 'returns'}, inplace=True)
data_fast.rename(columns={'synt_returns': 'returns'}, inplace=True)

aap_data_5 = data_fast.copy()
aap_data_10 = data_slow.copy()

ohlc.index = pd.to_datetime(ohlc.index)
aap_data_5.index = pd.to_datetime(aap_data_5.index)
aap_data_10.index = pd.to_datetime(aap_data_10.index)

aap_data_5.dropna(inplace=True)
aap_data_10.dropna(inplace=True)

common_indexes = [i for i in aap_data_10.index if i in aap_data_5.index and i in ohlc.index]

aap_data_10 = aap_data_10.loc[common_indexes, :]
aap_data_5 = aap_data_5.loc[common_indexes, :]
ohlc = ohlc.loc[common_indexes, :]

# aap_data_5.drop(aap_data_5.columns[-8:-1], axis = 1,inplace = True)
# aap_data_10.drop(aap_data_10.columns[-8:-1], axis = 1,inplace = True)

# aap_data_10 = aap_data_10.resample("15Min").mean()
# aap_data_5 = aap_data_5.resample("15Min").mean()
# ohlc = ohlc.resample("15Min").mean()
#
# aap_data_10.interpolate(inplace=True)
# aap_data_5.interpolate(inplace=True)
# ohlc.interpolate(inplace=True)

parameters_file = open("trained_parameters.pkl", "rb")
parameters = pickle.load(parameters_file)
parameters_file.close()



test_env = CustomEnv(ohlc, aap_data_5, aap_data_10, Render_range=500, initial_balance=2000, base_bet=1500,
                      retraining=20 * 48, training_size=training_size, lookback_window_size=lookback_window_size,
                      prediction_horizon=(10, 80))

trades_list, worth_evolution, returns, vola_ds, vola_ps = Play_env(parameters, test_env, start_render=training_size,
                                                                                visualize=False, train_episodes=1)

worth_evolution = pd.DataFrame(worth_evolution, index=ohlc.index[-len(worth_evolution):])
# returns = pd.DataFrame(returns, index=ohlc_5.index[-len(returns):])

plt.figure(3)
for i in range(worth_evolution.shape[0]):
    plt.plot(ohlc.index[-worth_evolution.shape[1]:], worth_evolution.iloc[i, :])
plt.title('Portfolio Value Evolution')
plt.show()

trades_list = pd.DataFrame(trades_list)
trades_list['timo signal'] = [1 if (trades_list['type'][i] == 'buy' or
                                    'Short Closed' in trades_list['flag'][i] or
                                    'Short Take Profit' in trades_list['flag'][i] or
                                    'Short Stop Loss' in trades_list['flag'][i]) else -1 for i in
                              range(len(trades_list))]
#
# trades_list.to_csv('aml_workspace/notebooks/research_topics/stefano/backtester/trades_list_ls.csv')
# returns.to_csv('aml_workspace/notebooks/research_topics/stefano/backtester/strategy_returns_ls.csv')


ds_ps = pd.DataFrame([vola_ds, vola_ps]).transpose()
ds_ps.columns = ['ds', 'ps']
ds_ps = ds_ps.loc[~(ds_ps == 0).all(axis=1)]
we = worth_evolution.T
ds_ps['worth'] = we

