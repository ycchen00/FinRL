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
# 4 pre_train
exact_model_name = None  # "3s"#None#"final"  # ""

pre_model_file_name = "dqn_AAPL_multi_u23_t20w_r20rr10hr1ce4tr3_2021-0606-1554"

# 4 env
r_negative_minus = 20
rr_positive_mul = 20
hr_mul = 1
ce_positive_mul = 8
tr_bm_coef = 2

# 4 model
model_name = pre_model_file_name[:3] # 'dqn'
if model_name == "dqn":
    model_kwargs = {
        "batch_size": 32,
        "buffer_size": 100000,  # 1000000,
        "learning_rate": 0.0001}
else:
    pass

# 4 train
timesteps = 0.01
per_save_timesteps = 15
early_stopping_patience = None

# 4 predict
predict_train = False


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


##########################################################################
external_info = f"r{r_negative_minus}rr{rr_positive_mul}hr{hr_mul}_retrain"

env_kwargs = {
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": TECHNICAL_USERDEFINED_LIST,
    "action_space": stock_dimension,
    "r_negative_minus": r_negative_minus,
    "rr_positive_mul": rr_positive_mul,
    "hr_mul": hr_mul,
    "ce_positive_mul": ce_positive_mul,
    "tr_bm_coef": tr_bm_coef
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

# model_file_name = f"{model_name}_{tic}_multi_u{state_space - 1}_t{timesteps}w_{external_info}_{datetime.datetime.now().strftime('%Y-%m%d-%H%M')}"
model_file_name = 'dqn_AAPL_multi_u23_t1w_r20rr20hr1_retrain_2021-0609-0805'
models_path = f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{model_file_name}"
result_path = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}/{model_file_name}"

agent = DRLAgent(env=env_train)
model_ = agent.get_model(model_name, model_kwargs=model_kwargs)

model_.save(f"{models_path}/init")
# predict_kwargs = {
#     "model_": model_,
#     "models_path": models_path,
#     "result_path": f"{result_path}/init",
#     "model_file_name": pre_model_file_name,
# }
# Predict
# predict_save_models(
#     e_trade_gym=e_trade_gym,
#     e_train_gym=e_train_gym,
#     predict_train=predict_train,
#     exact_model_name='init',
#     **predict_kwargs)


# Load
env_train.reset()
if exact_model_name is None:
    exact_model_name = 'final'
model_ = model_.load(f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{pre_model_file_name}/{exact_model_name}",env_train)
env_train.reset()
# predict_kwargs = {
#     "model_": model_,
#     "models_path": f"/home/cyc/RL_STOCK/FinRL-multi/{config.TRAINED_MODEL_DIR}/{pre_model_file_name}",
#     "result_path": f"{result_path}/pre_trained",
#     "model_file_name": pre_model_file_name,
# }
# # Predict
# predict_save_models(
#     e_trade_gym=e_trade_gym,
#     e_train_gym=e_train_gym,
#     predict_train=predict_train,
#     exact_model_name='final',
#     **predict_kwargs)


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
    exact_model_name='final',
    **predict_kwargs)

print(f'**Success train and predict of {model_file_name}**\n')
pass