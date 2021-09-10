import numpy as np
import yfinance
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import logger

matplotlib.use("Agg")


def _cal_exp(x, ind=2.0):
    return abs(x) ** ind * (-1) ** int(x < 0)


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            df,
            stock_dim,
            state_space,
            action_space,
            tech_indicator_list,
            buy_cost_pct=0,
            sell_cost_pct=0,
            turbulence_threshold=None,
            make_plots=False,
            print_verbosity=10,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
            initial_amount=10000,
            stepmoney=10000,
            # transaction_cost = 0.015,
            transaction_cost=0.07,
            time_range_benchmark=10,
            reward_scaling=1,
            # reward
            ind=3,
            rt_scaling=1000,
            roi_threshold=0.0,
            rt_positive_minus=0,
            rt_negative_minus=1,
            # timeb_positive_scaling=1,
            # timeb_negative_scaling=2,
            ratio_threshold=0.2,
            _b_add=1,
            _b_ind=2,
            _a_add=1,
            _a_mul=1.5,
            r_negative_minus=10,  # 0 - 50
            rr_positive_mul=30,  # 0 - 50
            ce_positive_mul=1,
            tr_bm_coef=2,
            # holding reward
            latest_day=30,
            time_range_allowed=7,
            time_range_bm=20,
            ratio_base=0.05,
            ratio_bm=0.07,
            hr_mul=1,  # 1- 100
    ):
        self.day = day
        self.df = df
        self.close = df[['close', 'tic']]
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.end_total_asset = initial_amount
        self.end_total_account = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,))
        # self.data = self.df.loc[self.day, :]
        self.data = self.df[self.df.tic ==
                            self.df.tic.unique()[0]].loc[self.day]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.stepmoney = stepmoney
        self.transaction_cost = transaction_cost
        self.time_range_benchmark = time_range_benchmark
        self.fr_day = 10

        # initializes
        self.state = self._initiate_state()
        self.trade_cache = self._initiate_trade_cache()
        self.trade_state_memory = self._initiate_trade_state_memory()
        self.empty_state_memory = self._initiate_empty_state_memory()
        # self.trade_data_list = []
        self.buy_day = -1
        self.sell_day = -1
        self.tic = None
        self.trade_index = 0
        self.reward = 0

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        # self.rewards_memory = []
        # self.actions_memory = []
        # self.states_memory = []
        self.date_memory = [self._get_date()]
        self.reset()
        self._seed()

        # reward pars
        self.ind = ind
        self.rt_scaling = rt_scaling  # 1000
        self.roi_threshold = roi_threshold  # 0.0
        self.ratio_threshold = ratio_threshold
        self.rt_positive_minus = rt_positive_minus  # 0
        self.rt_negative_minus = rt_negative_minus  # 1
        self._b_add = _b_add
        self._b_ind = _b_ind
        self._a_add = _a_add
        self._a_mul = _a_mul
        self.tr_bm_coef = tr_bm_coef
        # self.timeb_positive_scaling = timeb_positive_scaling  # 1
        # self.timeb_negative_scaling = timeb_negative_scaling  # 2

        self.r_negative_minus = r_negative_minus
        self.rr_positive_mul = rr_positive_mul
        self.ce_positive_mul = ce_positive_mul
        # holding reward
        self.latest_day = latest_day
        self.time_range_allowed = time_range_allowed
        self.time_range_bm = time_range_bm
        self.ratio_base = ratio_base
        self.ratio_bm = ratio_bm
        self.hr_mul = hr_mul

    def _sell_stock(self):
        def _do_sell_normal():
            close_price = self.close[self.close['tic']
                                     == self.tic].iloc[self.day]['close']
            if close_price > 0:
                # Sell only if the price is > 0 (no missing data in this
                # particular date)
                if self.buy_day > -1 and self.sell_day == -1 and self.day != self.buy_day:
                    # Sell only if sell_day != -1  and it is not sold yet
                    buy_timestamp = int(self.buy_day)
                    sell_num_shares = self.stepmoney // self.close[self.close['tic']
                                                                   == self.tic].iloc[buy_timestamp]['close']
                    sell_amount = close_price * \
                                  sell_num_shares * (1 - self.sell_cost_pct)
                    # update balance
                    self.end_total_account += sell_amount

                    self.cost += close_price * \
                                 sell_num_shares * self.sell_cost_pct
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares

        close_price = self.close[self.close['tic']
                                 == self.tic].iloc[self.day]['close']
        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if close_price > 0:
                    # # Sell only if the price is > 0 (no missing data in this particular date)
                    # # if turbulence goes over threshold, just clear out all positions
                    sell_num_shares = _do_sell_normal()
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()
        if sell_num_shares != 0:
            self.sell_day = self.day
        else:
            self.sell_day = -1
        return sell_num_shares

    def _buy_stock(self):
        def _do_buy():
            close_price = self.close[self.close['tic']
                                     == self.tic].iloc[self.day]['close']
            if close_price > 0:
                # Buy only if the price is > 0 (no missing data in this
                # particular date)
                if self.trade_index <= -1 and self.buy_day == -1 and self.sell_day == -1:
                    available_amount = self.stepmoney // close_price

                    # update balance
                    buy_num_shares = available_amount
                    buy_amount = close_price * \
                                 buy_num_shares * (1 + self.buy_cost_pct)
                    self.end_total_account -= buy_amount

                    self.cost += close_price * \
                                 buy_num_shares * self.buy_cost_pct

                    self.buy_day = self.day
                    self.trades += 1
                    self.trade_index = self.trades
                else:
                    buy_num_shares = 0
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass
        # self.states_buy.append(self.day)

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, 'r')
        plt.savefig('results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def _get_close(self, timestamp):
        if timestamp == -1:
            return 0
        return self.df[self.df.tic == self.tic].loc[timestamp].close

    def steps(self, action, state):
        self.state = state
        self.trade_index = int(state[0])

        trade_data = self.trade_cache[self.trade_cache['trade_index']
                                      == self.trade_index]
        try:
            self.day = int(trade_data.today)
            self.buy_day = int(trade_data.buy_day)
            self.sell_day = int(trade_data.sell_day)
            self.tic = trade_data.tic.values[0]
            self.terminal = self.day >= len(self.df.index.unique()) - 1
        except BaseException:
            # print('self.trade_data_list:', self.trade_data_list)
            print('self.trade_index', self.trade_index)
            print('failed')

        if self.terminal:
            print('terminal')
            return self.state, self.reward, action, self.terminal, {}
        else:
            temp = action
            # if self.stock_dim == 1:
            if action == 0:
                action_num = self._sell_stock()
            elif action == 2:
                action_num = self._buy_stock()
            else:
                action_num = 0

            if action_num == 0:
                action = 1

            # just 4 test
            # self.reward = self._get_reward()
            # buy_price = self.get_close(self.buy_day)
            # sell_price = self.get_close(self.sell_day)
            # today_price = self.get_close(self.day)
            # print(f"state:{self.state[:4]}\tb4action:{temp}\taction:{action}")
            # print(f"buy_day:{self.buy_day}\tsell_day:{self.sell_day}\ttoday:{self.day}")
            # print(f"buy_price:{buy_price}\tsell_price:{sell_price}\ttoday_price:{today_price}")

            self.day += 1
            self.data = self.df[self.df.tic == self.tic].loc[self.day]

            self.date_memory.append(self._get_date())
            self.reward = self._get_reward()
            # self.asset = self._get_asset()
            # self.reward = self.reward * self.reward_scaling

            self._update_trade_cache()
            # update state
            self.state = self._update_state()

            self.state[-10:] = np.random.rand(10)

            if self.state[0] >= 0:  # we have bought
                # feature_rb
                self._update_state_rb()
                # feature_rh
                self.state[-10:-5] = [0] * 5
                # self.state[-10:-5] = self.state[-5:]
                # print(self.state[-5:])
            elif self.day > self.fr_day:  # not buy yet
                # feature_rb
                self.state[-5:] = [0] * 5
                # feature_rh
                self._update_state_rh()
                # print(self.state[-10:-5])

            for i in range(1, 11):
                j = -i
                v = self.state[j]
                if v > 0:
                    self.state[j] = 1
                elif v < 0:
                    self.state[j] = -1

            self._update_trade_state_memory()
            self._update_empty_state_memory()

            # print(f"next state:{self.state[:4]}\treward:{self.reward}\n")
            # TODO: not modified yet
            # if self.buy_day != -1:
            #     buy_timestamp = int(self.buy_day)
            #     market_asset = self.stepmoney - \
            #         self.stepmoney % self.close[self.close['tic'] == self.tic].iloc[buy_timestamp]['close']
            # else:
            #     market_asset = 0
            #
            # end_total_asset = self.end_total_account + market_asset
            #
            # self.asset_memory.append(end_total_asset)

        return self.state, self.reward, action, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()
        self.trade_cache = self._initiate_trade_cache()
        self.trade_state_memory = self._initiate_trade_state_memory()
        self.empty_state_memory = self._initiate_empty_state_memory()

        if self.initial:
            # TODO: update asset_memory or delete it?
            self.asset_memory = [self.initial_amount]

        self.day = 0
        self.data = self.df[self.df.tic ==
                            self.df.tic.unique()[0]].loc[self.day]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        # self.rewards_memory = []
        # self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self, mode='human', close=False):
        return self.state

    def _initiate_state(self):
        tics = self.df.tic.unique()
        if self.initial:
            # For Initial State
            if len(tics) == 1:
                tic = tics[0]
                self.data = self.df[self.df.tic == tic].loc[0]
                state = (
                        [-1]  # trade index
                        + sum([[self.data[tech]]
                               for tech in self.tech_indicator_list], [])
                )

                return state
            else:
                states = []
                i = -len(tics)
                for tic in tics:
                    self.data = self.df[self.df.tic == tic].loc[0]
                    state = (
                            [i]  # trade index
                            + sum([[self.data[tech]]
                                   for tech in self.tech_indicator_list], [])
                    )
                    states.append(state)
                    i += 1
                return states
        else:
            # Using Previous State
            state = ([self.previous_state[0]] + sum([[self.data[tech]]
                                                     for tech in self.tech_indicator_list], []))
            return state

    def _cal_reward(self, current_timestamp, buy_timestamp):
        # TODO: update for multi stocks

        buy_price = self._get_close(buy_timestamp)
        current_price = self._get_close(current_timestamp)

        ratio = (current_price - buy_price) / buy_price
        time_range = current_timestamp - buy_timestamp
        roi = min(ratio - self.transaction_cost, self.ratio_threshold)

        _b = _cal_exp(self._b_add + roi, self._b_ind)
        _a = self._a_add + ratio / self.ind * self._a_mul

        if roi > self.roi_threshold:
            # time_base = min(max(time_range, self.time_range_bm), self.time_range_bm * self.tr_bm_coef)
            time_base = min(max(time_range, self.time_range_bm), self.time_range_bm ** (4/5))
            return _cal_exp(
                (roi / time_base * 100 * 2 + 1) * _a,
                self.ind * _b * _a) * self.ce_positive_mul + ratio * self.rr_positive_mul
            # return min(500,reward)
        else:
            time_base = max(
                min(time_range, self.time_range_bm / 2), self.time_range_bm / 4)
            return _cal_exp(
                roi / time_base * 100 - self.rt_negative_minus,
                self.ind) - self.r_negative_minus

    def _holding_reward(self):
        current_timestamp = int(self.day)
        latest_sell_day = max(self.trade_cache.sell_day)
        if latest_sell_day == -1:
            return 0

        start_timestamp = max(
            latest_sell_day,
            current_timestamp -
            self.latest_day)

        start_price = self._get_close(start_timestamp)
        current_trend = self.df[self.df.tic ==
                                self.tic].loc[current_timestamp].trend
        current_close = self._get_close(current_timestamp)

        current_price = min(current_close, current_trend)
        ratio = int((current_price - start_price) /
                    start_price)
        time_range = current_timestamp - start_timestamp
        roi = ratio - self.transaction_cost / 2
        if abs(roi) < self.transaction_cost * 1.5:
            return 0
        elif roi < self.transaction_cost * 2:
            if time_range < self.time_range_bm:
                return 0
        return (- roi * _cal_exp(time_range, 0.5)) * self.hr_mul

    def _get_reward(self):
        if self.buy_day == -1:
            reward = self._holding_reward()
        elif self.sell_day != -1:
            buy_timestamp = int(self.buy_day)
            sell_timestamp = int(self.sell_day)

            reward = self._cal_reward(sell_timestamp, buy_timestamp)
        else:
            buy_timestamp = int(self.buy_day)
            current_timestamp = int(self.day)

            reward = self._cal_reward(current_timestamp, buy_timestamp)
        return reward


    def _simple_reward(self, current_timestamp, buy_timestamp):
        buy_price = self._get_close(buy_timestamp)
        current_price = self._get_close(current_timestamp)
        reward = current_price / buy_price - 1
        return reward

    def _update_state_rh(self):
        current_timestamp = int(self.day)
        buy_timestamp = int(self.day - self.fr_day)
        for i in range(5):
            self.state[-6 - i] = self._cal_reward(current_timestamp - i,
                                                  buy_timestamp) - self._cal_reward(current_timestamp - i - 1,
                                                                                    buy_timestamp)

    def _update_state_rb(self):
        buy_timestamp = int(self.buy_day)
        current_timestamp = int(self.day)
        backtrack_timestamp = current_timestamp - buy_timestamp - 1
        if backtrack_timestamp >= 5:
            for i in range(5):
                self.state[-i - 1] = self._cal_reward(current_timestamp - i,
                                                      buy_timestamp) - self._cal_reward(current_timestamp - i - 1,
                                                                                        buy_timestamp)
        elif backtrack_timestamp == 0:
            self.state[-5:] = [0] * 5
        else:
            self.state[-5:] = [0] * 5
            for i in range(backtrack_timestamp):
                # self.state[-i - 1] = self.rewards_memory[current_timestamp - \
                #     i - 1] - self.rewards_memory[current_timestamp - i - 2]
                self.state[-i - 1] = self._cal_reward(current_timestamp - i,
                                                      buy_timestamp) - self._cal_reward(current_timestamp - i - 1,
                                                                                        buy_timestamp)

    def _update_state(self):
        state = ([self.trade_index] + sum([[self.data[tech]]
                                           for tech in self.tech_indicator_list], []))
        return state

    def _initiate_trade_cache(self):
        # if self.initial:
        tic_names = self.df.tic.unique()
        trade_cache = pd.DataFrame({
            'trade_index': range(-len(tic_names), 0),  # [-1] * len(tic_names),
            'today': [0] * len(tic_names),
            'buy_day': [-1] * len(tic_names),
            'sell_day': [-1] * len(tic_names),
            'buy_price': [0] * len(tic_names),
            'sell_price': [0] * len(tic_names),
            'ratio': [0] * len(tic_names),
            'day_range': [0] * len(tic_names),
            'daily_ratio': [0] * len(tic_names),
            'reward': [0] * len(tic_names),
            'asset': [0] * len(tic_names),
            'tic': tic_names
        })
        return trade_cache

    def _update_trade_cache(self):
        if len(
                self.trade_cache[self.trade_cache['trade_index'] == self.trade_index]) > 0:
            # if such trade_ibndex exists == sell / hold
            if self.trade_index >= 0:
                buy_price = self._get_close(self.buy_day)
                if self.sell_day != -1:
                    sell_price = self._get_close(self.sell_day)
                    day_range = self.sell_day - self.buy_day
                else:
                    sell_price = self._get_close(self.day)
                    day_range = self.day - self.buy_day
                ratio = sell_price / buy_price - 1
                daily_ratio = ratio / day_range
                asset = sell_price / buy_price * self.stepmoney
            else:
                buy_price = 0
                sell_price = 0
                day_range = 0
                ratio = 0
                daily_ratio = 0
                asset = 0
            self.trade_cache[
                (self.trade_cache['trade_index'] == self.trade_index) & (self.trade_cache['tic'] == self.tic)] = [
                self.trade_index, self.day, self.buy_day, self.sell_day, buy_price, sell_price, ratio, day_range,
                daily_ratio, self.reward, asset, self.tic]
        else:
            # new trade_index : after buy
            buy_price = self.df[self.df.tic == self.tic].loc[self.day].close
            sell_day = -1
            sell_price = 0
            day_range = 0
            ratio = 0
            daily_ratio = 0
            asset = 0
            self.trade_cache.loc[len(self.trade_cache)] = [self.trade_index,
                                                           self.day,
                                                           self.day,
                                                           sell_day,
                                                           buy_price,
                                                           sell_price,
                                                           ratio,
                                                           day_range,
                                                           daily_ratio,
                                                           self.reward,
                                                           asset,
                                                           self.tic]
        pass

    def _initiate_trade_state_memory(self):
        # if self.initial:
        initial_state = self._initiate_state()
        tic_names = self.df.tic.unique().tolist()
        if len(tic_names) == 1:
            trade_state_memory = pd.DataFrame({
                'trade_index': [-1],
                'day': [0],
                'state': [initial_state],
                'reward': [0],
                'asset': [0],
                'tic': tic_names[0]
            })
        else:
            trade_state_memory = pd.DataFrame({
                # [-1] * len(tic_names),
                'trade_index': range(-len(tic_names), 0),
                'day': [0] * len(tic_names),
                'state': initial_state,
                'reward': [0] * len(tic_names),
                'asset': [0] * len(tic_names),
                'tic': tic_names
            })
        return trade_state_memory

    def _update_trade_state_memory(self):
        # add one
        if self.trade_index >= 0:
            if self.sell_day == -1:
                buy_price = self._get_close(self.buy_day)
                sell_price = self._get_close(self.day)
                asset = sell_price / buy_price * self.stepmoney
                self.trade_state_memory.loc[len(self.trade_state_memory)] = [
                    self.trade_index, self.day, self.state, self.reward, asset, self.tic]

    def _initiate_empty_state_memory(self):
        # if self.initial:
        initial_state = self._initiate_state()
        tic_names = self.df.tic.unique().tolist()
        if len(tic_names) == 1:
            trade_state_memory = pd.DataFrame({
                'day': [0],
                'state': [initial_state],
                'reward': [0],
                'tic': tic_names[0]
            })
        else:
            trade_state_memory = pd.DataFrame({
                'day': [0] * len(tic_names),
                'state': initial_state,
                'reward': [0] * len(tic_names),
                'tic': tic_names
            })
        return trade_state_memory

    def _update_empty_state_memory(self):
        # add one
        if self.trade_index <= -1:
            self.empty_state_memory.loc[len(self.empty_state_memory)] = [
                self.day, self.state, self.reward, self.tic]

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date  # .unique()[0]
        else:
            date = self.data.date
        return date

    def save_final_asset(self):
        asset = self.trade_cache['asset']
        filter_asset = asset[asset > 0]
        return sum(filter_asset), len(filter_asset)

    def save_trade_cache(self):
        return self.trade_cache

    def save_trade_state_memory(self):
        return self.trade_state_memory

    def save_empty_state_memory(self):
        return self.empty_state_memory

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {'date': date_list, 'account_value': asset_list})
        return df_account_value

    # def save_action_memory(self):
    #     if len(self.df.tic.unique()) > 1:
    #         # date and close price length must match actions length
    #         date_list = self.date_memory[:-1]
    #         df_date = pd.DataFrame(date_list)
    #         df_date.columns = ['date']
    #
    #         action_list = self.actions_memory
    #         df_actions = pd.DataFrame(action_list)
    #         df_actions.columns = self.data.tic.values
    #         df_actions.index = df_date.date
    #         #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
    #     else:
    #         date_list = self.date_memory[:-1]
    #         action_list = self.actions_memory
    #         df_actions = pd.DataFrame(
    #             {'date': date_list, 'actions': action_list})
    #     return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
