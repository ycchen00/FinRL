import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import trade_plot
import re
from test_packages import *
import sys
sys.path.append("../FinRL-multi")

############################## Changed pars ##############################
# 4 predict
predict_train = False
exact_model_name = None  # "3s"#None#"final"  # ""

model_file_names = [  # "a2c_AAPL_multi_u23_t20w_r0rr20hr20_2021-0530-0433",
    # "a2c_AAPL_multi_u23_t20w_r10rr0hr5_2021-0528-0639",
    # "a2c_AAPL_multi_u23_t20w_r10rr10hr20_2021-0529-0648",
    # "a2c_AAPL_multi_u23_t20w_r50rr20hr10_2021-0530-1912",
    # "a2c_AAPL_multi_u23_t20w_r70rr10hr5_2021-0529-2113"
    "dqn_AAPL_multi_u23_t1w_r20rr20hr1_2021-0602-1012"]


############################## Unchanged pars ##############################
# 4 data
flag_pro = True
tic = "AAPL"
TECHNICAL_USERDEFINED_LIST = config.TECHNICAL_USERDEFINED_LIST_23
stock_dimension = 1
state_space = len(TECHNICAL_USERDEFINED_LIST) + 1
processed_full = get_processed_full(tic, state_space, flag_pro)

tics = processed_full.tic.unique().tolist()
train = data_split(processed_full, "2009-01-01", "2019-01-01")
trade = data_split(processed_full, "2019-01-01", "2021-01-01")

# 4 model
# model_name = 'dqn'
model_kwargs = {
    "batch_size": 32,
    "buffer_size": 100000,  # 1000000,
    "learning_rate": 0.0001}

show_details = False
if show_details:
    print('Dataset info:')
    print(f'trainset len: {len(train)}')
    print(f'tradeset len: {len(trade)}\n')
    print(f'Model kwargs:\n{model_kwargs}')

# 4 train
timesteps = 1
per_save_timesteps = 15
early_stopping_patience = None


##########################################################################
for model_file_name in model_file_names:
    models_path = f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{model_file_name}"
    result_path = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}/{model_file_name}"
    model_name = model_file_name[:3]

    # 4 env
    try:
        r_negative_minus = re.search(r'_r([\d]+)\S+', model_file_name).group(1)
        rr_positive_mul = re.search(r'\S+rr([\d]+)\S+', model_file_name).group(1)
        hr_mul = re.search(r'\S+hr([\d]+)\S+', model_file_name).group(1)
    except:
        print(f'~~{model_file_name} no r_rr_hr pars, ignore~~')
        continue

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
    agent = DRLAgent(env=env_train)
    model_ = agent.get_model(model_name, model_kwargs=model_kwargs)

    predict_kwargs = {
        "model_": model_,
        "models_path": models_path,
        "result_path": result_path,
        "model_file_name": model_file_name,
    }

    predict_save_models(
        e_trade_gym=e_trade_gym,
        e_train_gym=e_train_gym,
        predict_train=predict_train,
        exact_model_name=exact_model_name,
        **predict_kwargs)
