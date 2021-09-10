import pandas as pd
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline,trx_plot

from pprint import pprint

import sys

sys.path.append("../FinRL_bsbs")

import itertools
import os
import matplotlib.dates as mdates


df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/data_AAPL.csv")
df.drop(["Unnamed: 0"], axis=1, inplace=True)
df.sort_values(["date", "tic"], ignore_index=True)

fe = FeatureEngineer(
    use_technical_indicator=False,
    # tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    user_defined_feature=True,#config.TECHNICAL_USERDEFINED_LIST,
    # use_turbulence=True,
    # user_defined_feature=False,
)

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])

processed_full = processed_full.fillna(0)

processed_full.sort_values(["date", "tic"], ignore_index=True)

train = data_split(processed_full, "2009-01-01", "2019-01-01")
trade = data_split(processed_full, "2019-01-01", "2021-01-01")


stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(config.TECHNICAL_USERDEFINED_LIST) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")



processed_full.to_csv("./" + config.DATA_SAVE_DIR + "/data_AAPL_user8.csv")
# env_kwargs = {
#     "hmax": 100,
#     "initial_amount": 10000,
#     "buy_cost_pct": 0.001,
#     "sell_cost_pct": 0.001,
#     "state_space": state_space,
#     "stock_dim": stock_dimension,
#     # "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
#     "tech_indicator_list": config.TECHNICAL_USERDEFINED_LIST,
#     "action_space": stock_dimension,
#     "reward_scaling": 1e-4,
# }
#
# e_train_gym = StockTradingEnv(df=train, **env_kwargs)
#
# env_train, _ = e_train_gym.get_sb_env()
#
# agent = DRLAgent(env=env_train)
# model_name="a2c"
# model_a2c = agent.get_model(model_name)
#
#
# # trained_a2c = agent.train_model(model=model_a2c, tb_log_name=model_name, total_timesteps=50000)
#
#
# # trained_a2c.save("./"+config.TRAINED_MODEL_DIR+"/a2c_AAPL_"+datetime.datetime.now().strftime('%Y%m%d'))
# trained_a2c=model_a2c.load("./"+config.TRAINED_MODEL_DIR+"/a2c_AAPL_20210428.zip")
#
#
# e_trade_gym = StockTradingEnv(df = trade,  **env_kwargs)
#
# df_account_value, df_actions = DRLAgent.DRL_prediction(
#     model=trained_a2c,
#     environment = e_trade_gym)
#
# trx_plot(trade, df_actions, ['AAPL'])
# plt.show()
# # plt.savefig('./test5.jpg')

print("succeed")
