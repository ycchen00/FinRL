import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline, trx_plot, trade_plot
from test_packages import *
from pprint import pprint
import sys
sys.path.append("../FinRL-multi")

from test_packages import *
############################## Unchanged pars ##############################
# 4 data
flag_pro = True
tic = "30S"
TECHNICAL_USERDEFINED_LIST = config.TECHNICAL_USERDEFINED_LIST_23
stock_dimension = 1
state_space = len(TECHNICAL_USERDEFINED_LIST) + 1
processed_full = get_processed_full(tic, state_space, flag_pro)

tics = processed_full.tic.unique().tolist()
train = data_split(processed_full, "2009-01-01", "2019-01-01")
train = data_split(processed_full, "2009-01-01", "2010-01-01")
trade = data_split(processed_full, "2019-01-01", "2021-01-01")

print('Dataset info:')
print(f'trainset len: {len(train)}')
print(f'tradeset len: {len(trade)}\n')


############################## Changed pars ##############################
# 4 env
r_negative_minus = 20
rr_positive_mul = 20
hr_mul = 1

# 4 model
model_name = 'a2c'
model_kwargs = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}

print(f'Model kwargs:\n{model_kwargs}')

# 4 train
timesteps = 1
per_save_timesteps = 15
early_stopping_patience = None

# 4 predict
predict_train = True


##########################################################################
external_info = f"r{r_negative_minus}rr{rr_positive_mul}hr{hr_mul}"

env_kwargs = {
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": TECHNICAL_USERDEFINED_LIST,
    "action_space": stock_dimension,
    "r_negative_minus": r_negative_minus,
    "rr_positive_mul": rr_positive_mul,
    "hr_mul": hr_mul
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

model_file_name = f"{model_name}_{tic}_multi_u{state_space - 1}_t{timesteps}w_{external_info}_{datetime.datetime.now().strftime('%Y-%m%d-%H%M')}"
models_path = f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{model_file_name}"
result_path = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}/{model_file_name}"

agent = DRLAgent(env=env_train)
model_ = agent.get_model(model_name, model_kwargs=model_kwargs)
train_kwargs = {
    "model_": model_,
    "timesteps": timesteps,
    "model_name": model_name,
    "models_path": models_path,
    "model_file_name": model_file_name,
    "per_save_timesteps": per_save_timesteps,
    "early_stopping_patience": early_stopping_patience
}
predict_kwargs = {
    "model_": model_,
    "models_path": models_path,
    "result_path": result_path,
    "model_file_name": model_file_name,
}


##########################################################################
# Train
trained_ = train_save_models(agent, **train_kwargs)

# Predict
predict_save_models(
    e_trade_gym=e_trade_gym,
    e_train_gym=e_train_gym,
    predict_train=predict_train,
    exact_model_name=None,
    **predict_kwargs)

print(f'**Success train and predict of {model_file_name}**\n')
pass
