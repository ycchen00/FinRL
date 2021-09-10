import os
import pandas as pd
import numpy as np
import re
import time
import datetime
import sys
from finrl.config import config
sys.path.append("../FinRL-multi")


def _preprocess_day_range(trade_cache):
    for i in range(len(trade_cache)):
        if trade_cache.iloc[i].trade_index == -1:
            continue
        if trade_cache.iloc[i].sell_day == -1:
            trade_cache.loc[i, 'day_range'] = trade_cache.iloc[i].today - \
                trade_cache.iloc[i].buy_day
    return trade_cache


def cal_profit(trade_cache):
    trade_cache = _preprocess_day_range(trade_cache)
    profit = {
        'ratio_range': 0.0,
        'asset': 0,
        'trade_num': 0,
        'total_ratio': 0,
        'ratio_num': 0}

    buy_price = trade_cache['buy_price']
    sell_price = trade_cache['sell_price']
    ratio = trade_cache['ratio']
    day_range = trade_cache['day_range']
    trade_num = len(buy_price[buy_price > 0])
    sum_day_range = int(sum(day_range))
    sum_ratio = float(sum(ratio))
    if sum_day_range <= 0:
        return profit
    profit['total_ratio'] = sum_ratio
    profit['ratio_range'] = sum_ratio / sum_day_range
    profit['ratio_num'] = sum_ratio / trade_num
    profit['trade_num'] = trade_num
    return profit


def timestamp2time(timestamp):
    try:
        time_struct = time.localtime(timestamp)
    except BaseException:
        print('error')
        pass
    return time.strftime('%Y%m%d', time_struct)


def get_file_access_time(filePath):
    t = os.path.getatime(filePath)
    return timestamp2time(t)


def get_file_create_time(filePath):
    t = os.path.getctime(filePath)
    return timestamp2time(t)


def get_file_modify_time(filePath):
    t = os.path.getmtime(filePath)
    return timestamp2time(t)


def init_csv(path,csv_name):
    if os.path.exists(f"{path}/{csv_name}.csv"):
        result_df = pd.read_csv(f"{path}/{csv_name}.csv",index_col=0)
        # result_df.model_name = result_df.model_name.astype(str)
    else:
        print(f'no find {csv_name}.csv, create one!')
        result_df = pd.DataFrame()
        result_df.to_csv(f"{path}/{csv_name}.csv")
    return result_df


def filter_filename(name, path, done_flag="done"):
    os.chdir(path)
    if os.path.isfile(name) or len(
            name) < 15 or get_file_create_time(name) < '20210527':
        print(f'【ignore unwanted file】:{name}')
        return True
    elif os.path.isdir(name):
        if os.path.exists(f"{path}/{name}/{done_flag}.txt"):
            print(f'【ignore done file】:{name}')
            return True
        return False
    else:
        print(f'【ignore special file】:{name}')
        return True


def extract_par(name, re_str, flag=0):
    if re_str == 'tr' and re.search('(trinf)', name) is not None:
        return np.inf

    if flag ==0:
        if re.search(rf'[\d_]{re_str}([\d]+)\S+', name) is None:
            return None
        return re.search(rf'[\d_]{re_str}([\d]+)\S+', name).group(1)
    else:
        if re.search(rf'{re_str}([\d]+)', name) is None:
            return None
        return re.search(rf'{re_str}([\d]+)', name).group(1)


def extract_pars(name, pars_names={},flag=0):
    pars = {}
    for full_name, re_str in pars_names.items():
        par = extract_par(name, re_str,flag)
        if par is None:
            print(f'【ignore no enough pars】:{name}')
            return False
        pars[full_name] = par
    return pars

def extract_Darwin(name,Darwin_names={},flag=0):
    pars = {}
    for full_name, re_str in Darwin_names.items():
        par = extract_par(name, re_str,flag)
        if par is None:
            print(f'【ignore no enough pars】:{name}')
            return False
        pars[full_name] = par
    return pars


