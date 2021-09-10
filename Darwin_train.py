import os
import time as tm
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline, trx_plot, trade_plot
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
import warnings
import pickle
from test_packages import *
from Darwin_packages import *
# from Darwin_script import *
print(0)
warnings.filterwarnings("ignore")
sys.path.append("../FinRL-multi")

os.chdir(f"{config.PROJECT_PATH}")

# ############### Argparse ###############
parser = argparse.ArgumentParser(description='')
parser.add_argument("-tkp", "--train_kwargs_path", help="path of train_kwargs")
parser.add_argument(
    "-pkp",
    "--predict_kwargs_path",
    help="path of predict_kwargs")
parser.add_argument("-ap", "--pars_dict_path", help="other args")
parser.add_argument("-mfn", "--model_file_name", help="model_file_name")
parser.add_argument(
    "-pmp",
    "--pre_model_path",
    help="pretrained model path",
    default=None)
# 4 predict
parser.add_argument(
    "-p_t",
    "--predict_train",
    help="whether predict train",
    type=bool,
    default=True)
args = parser.parse_args()
#
with open(args.pars_dict_path, 'rb') as handle:
    pars_dict = pickle.load(handle)
model_file_name = args.model_file_name
predict_train = args.predict_train

##########################################################################
# 4 data
flag_pro = True
tic = pars_dict["tic"]  # "AAPL"
if tic in ['AAPL','30S']:
    TECHNICAL_USERDEFINED_LIST = config.TECHNICAL_USERDEFINED_LIST_23
    stock_dimension = 1
    state_space = len(TECHNICAL_USERDEFINED_LIST) + 1
    processed_full = get_processed_full(tic, state_space, flag_pro)

    tics = processed_full.tic.unique().tolist()
    train = data_split(processed_full, "2009-01-01", "2019-01-01")
    trade = data_split(processed_full, "2019-01-01", "2021-01-01")
else:
    TECHNICAL_USERDEFINED_LIST = config.TECHNICAL_USERDEFINED_LIST_36
    stock_dimension = 1
    state_space = len(TECHNICAL_USERDEFINED_LIST) + 1
    processed_full = get_processed_full(tic, state_space, flag_pro)

    tics = processed_full.tic.unique().tolist()
    train, trade = data_split2(processed_full, 0.7)
# kwargs
model_kwargs = config.MODELS_PARAMS_DICT[pars_dict["model_name"]]
model_kwargs["device"]="cpu"


env_kwargs = {
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": TECHNICAL_USERDEFINED_LIST,
    "action_space": stock_dimension,
    "r_negative_minus": pars_dict["r_negative_minus"],
    "rr_positive_mul": pars_dict["rr_positive_mul"],
    "hr_mul": pars_dict["hr_mul"],
    "ce_positive_mul": pars_dict["ce_positive_mul"],
    "tr_bm_coef": pars_dict["tr_bm_coef"]
}

pars_kwargs = merge_dict(env_kwargs, {})

parallel_kwargs = {
    "n_parallel_task": pars_dict["n_parallel_task"],
    "m_per_timesteps": pars_dict["m_per_timesteps"],
    "M_tot_timesteps": pars_dict["M_tot_timesteps"],
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

agent = DRLAgent(env=env_train)
model_ = agent.get_model(pars_dict["model_name"], model_kwargs=model_kwargs)


if (args.pre_model_path is not None) and (args.pre_model_path != "None"):
    # /home/cyc/RL_STOCK/FinRL-multi/{config.TRAINED_MODEL_DIR}/{pre_model_file_name}/final
    model_ = model_.load(f"{args.pre_model_path}.zip", env_train)
    print(f"load {args.pre_model_path} succeeded!")
    env_train.reset()

Darwin_info = pars_dict['Darwin_info']#Darwin_str(pars_dict)
external_info = pars_dict['external_info']#pars_str(pars_kwargs)

train_kwargs, predict_kwargs = generate_tp_kwargs(
    Darwin_info, external_info, model_file_name, pars_dict)


# Train
trained_ = train_save_models(agent, model_=model_, **train_kwargs)

# Predict
predict_save_models(
    e_trade_gym=e_trade_gym,
    e_train_gym=e_train_gym,
    predict_train=predict_train,
    exact_model_name=None,
    model_=model_,
    **predict_kwargs)

print(f'**Success train and predict of {model_file_name}**\n')
