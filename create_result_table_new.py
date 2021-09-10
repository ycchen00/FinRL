import os
import pandas as pd
import numpy as np
import re
import time
import datetime
import sys
from finrl.config import config
sys.path.append("../FinRL-multi")
from analysis_packages import *


path = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}"

names = []
if os.path.exists(path):
    os.chdir(path)
    names = os.listdir()
    # length = len(names)
else:
    print('No such path!')

######################################
if os.path.exists(f"{path}/res_table.csv"):
    result_df = pd.read_csv(f"{path}/res_table.csv")
else:
    print(f'no find res_table.csv, create one!')
    result_df = pd.DataFrame()
######################################

done_flag = "done"
for name in names:
    # if name == '.idea' or name.endswith('.py') or name.endswith('.csv'):
    if filter_filename(name, path, done_flag): continue

    model = name[:3]
    steps = re.search(r'\S+t([\d]+\S)\S+', name).group(1)
    time_str = re.search(r'2021-([\S]+)', name).group(1)


    pars = extract_pars(name, config.PARS_NAMES)
    if not pars:
        continue
    # try:
    #     steps = re.search(r'\S+t([\d]+\S)\S+', name).group(1)
    #     r_negative_minus = re.search(r'[\d_]r([\d]+)\S+', name).group(1)
    #     rr_positive_mul = re.search(r'\S+rr([\d]+)\S+', name).group(1)
    #     hr_mul = re.search(r'\S+hr([\d]+)\S+', name).group(1)
    #     ce_positive_mul = re.search(r'\S+ce([\d]+)\S+', name).group(1)
    #     try:
    #         tr_bm_coef = re.search(r'\S+tr([\d]+)\S+', name).group(1)
    #     except:
    #         tr_bm_coef = np.inf
    #     time_str = re.search(r'2021-([\S]+)', name).group(1)
    #     pars = {
    #         'r_negative_minus': r_negative_minus,
    #         'rr_positive_mul': rr_positive_mul,
    #         'hr_mul': hr_mul,
    #         'ce_positive_mul': ce_positive_mul,
    #         'tr_bm_coef': tr_bm_coef}
    # except BaseException:
    #     print(f'【ignore no enough (r_rr_hr) pars】:{name}')
    #     continue

    files = os.listdir(name)
    if 'plot-backtest-final.jpg' not in files:
        print(f'【ignore not completed yet】:{name}')
        continue

    max_profit_backtest_info = ""
    min_profit_backtest_info = ""
    max_profit_backtest = -999999999
    min_profit_backtest = 999999999
    max_profit_train_info = ""
    min_profit_train_info = ""
    max_profit_train = -999999999
    min_profit_train = 999999999
    max_traden_train = 0
    min_traden_train = 0
    max_traden_backtest = 0
    min_traden_backtest = 0
    max_ratio_backtest = 0
    max_ration_backtest = 0
    max_ratio_train = 0
    max_ration_train = 0

    num_save = 0
    # step_money = 10000
    for file in files:
        if file.endswith('.txt'):
            if file.find('train-') == -1:  # backtest
                num_save += 1
                if 'final' in file:
                    step_model = 'final'
                else:
                    try:
                        step_model = re.search(r'(\d*s)-', file).group(1)
                    except:
                        step_model = re.search(
                            r'backtest-(\d*s)-', file).group(1)
                # trade_times = int(re.search('-\d*\.\d*_(\d+)', file).group(1))
                #             money = float(re.search('-(\d*\.\d*)_', file).group(1))
                trade_cache_name = f"trade_cache-backtest-{step_model}.csv"
                trade_cache_file = f"{name}/{trade_cache_name}"

                trade_cache = pd.read_csv(trade_cache_file, index_col=0)
                profit_info = cal_profit(trade_cache)
                ratio = profit_info['total_ratio']
                ratio_range = profit_info['ratio_range']
                ratio_num = profit_info['ratio_num']
                trade_num = profit_info['trade_num']

                # choose based on ratio_range
                if ratio_range > max_profit_backtest:
                    max_profit_backtest = ratio_range
                    max_profit_backtest_info = step_model
                    max_ratio_backtest = ratio
                    max_ration_backtest = ratio_num
                    max_traden_backtest = trade_num
                if ratio_range < min_profit_backtest:
                    min_profit_backtest = ratio_range
                    min_profit_backtest_info = step_model
                    # min_ratio_backtest = ratio
                    # min_ration_backtest = ratio_num
                    min_traden_backtest = trade_num
            else:  # train
                if file[:11] == 'train-final':
                    step_model = 'final'
                else:
                    step_model = re.search(r'train-(\d*s)-\S+', file).group(1)

                trade_cache_name = f"trade_cache-train-{step_model}.csv"
                trade_cache_file = f"{name}/{trade_cache_name}"

                trade_cache = pd.read_csv(trade_cache_file, index_col=0)
                profit_info = cal_profit(trade_cache)
                ratio = profit_info['total_ratio']
                ratio_range = profit_info['ratio_range']
                ratio_num = profit_info['ratio_num']
                trade_num = profit_info['trade_num']

                if ratio_range > max_profit_train:
                    max_profit_train = ratio_range
                    max_profit_train_info = step_model
                    max_ratio_train = ratio
                    max_ration_train = ratio_num
                    max_traden_train = trade_num
                if ratio_range < min_profit_train:
                    min_profit_train = ratio_range
                    min_profit_train_info = step_model
                    min_traden_train = trade_num
    maxs = max_profit_backtest * max_profit_train * 1000000
    criterion_1 = (max_profit_train * max_profit_backtest -
                   (max_profit_train - max_profit_backtest)**2) * 1000000
    criterion_2 = (max_ration_train * max_ration_backtest -
                   (max_ration_train - max_ration_backtest)**2) * 1000000

    info = pd.DataFrame({
        'name': [name],
        'model': [model],
        'steps': [steps],
        'pars': [f"{pars}"],
        'time': [time_str],
        'num_save': [num_save],

        'max_profit_backtest': [max_profit_backtest],
        'max_profit_backtest_info': [max_profit_backtest_info],
        'max_ratio_backtest': [max_ratio_backtest],
        'max_ration_backtest': [max_ration_backtest],
        'max_traden_backtest': [max_traden_backtest],

        'max_profit_train': [max_profit_train],
        'max_profit_train_info': [max_profit_train_info],
        'max_ratio_train': [max_ratio_train],
        'max_ration_train': [max_ration_train],
        'max_traden_train': [max_traden_train],

        'criterion_daily': [criterion_1],
        'criterion_times': [criterion_2],
        'maxs': [maxs],

        'min_profit_backtest': [min_profit_backtest],
        'min_profit_backtest_info': [min_profit_backtest_info],
        'min_traden_backtest': [min_traden_backtest],

        'min_profit_train': [min_profit_train],
        'min_profit_train_info': [min_profit_train_info],
        'min_traden_train': [min_traden_train],
    })

    result_df = pd.concat([result_df, info], axis=0)

# result_df['maxs'] = result_df['max_profit_backtest'] * \
#     result_df['max_profit_train'] * 1000000

# TODO: update instead of create a new one
result_df_name = f"res_table_{datetime.datetime.now().strftime('%Y-%m%d-%H%M')}.csv"
print(f"save {result_df_name}")
result_df.to_csv(result_df_name)
