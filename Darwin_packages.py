import os
import time as tm
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
import pickle
import random
from test_packages import *
from analysis_packages import *

warnings.filterwarnings("ignore")
sys.path.append("../FinRL-multi")
os.chdir(f"{config.PROJECT_PATH}")


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def pars_str(pars_kwargs):
    external_info = ""
    for key, value in pars_kwargs.items():
        if key in config.PARS_NAMES:
            external_info += f"{config.PARS_NAMES[key]}{value}"
    return external_info


def Darwin_str(pars_dict):
    Darwin_info = f'{pars_dict["model_name"]}_{pars_dict["tic"]}_n{pars_dict["n_parallel_task"]}m{pars_dict["m_per_timesteps"]}M{pars_dict["M_tot_timesteps"]}_{datetime.datetime.now().strftime("%Y-%m%d-%H%M")}'
    return Darwin_info


def get_models_path(Darwin_info, external_info, model_file_name):
    # Darwin_info = Darwin_str(pars_dict)
    models_path_1 = f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{Darwin_info}"
    models_path_2 = f"{models_path_1}/{external_info}"
    models_path_3 = f"{models_path_2}/{model_file_name}"

    if not os.path.exists(models_path_3):
        os.makedirs(models_path_3)
    return models_path_3, models_path_2


def get_result_path(Darwin_info, external_info, model_file_name):
    result_path_1 = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}/{Darwin_info}"
    result_path_2 = f"{result_path_1}/{external_info}"
    result_path_3 = f"{result_path_2}/{model_file_name}"

    if not os.path.exists(result_path_3):
        os.makedirs(result_path_3)
    return result_path_3, result_path_2


def generate_tp_kwargs(Darwin_info, external_info, model_file_name, pars_dict):
    models_path, _ = get_models_path(
        Darwin_info, external_info, model_file_name)
    result_path, _ = get_result_path(
        Darwin_info, external_info, model_file_name)

    train_kwargs = {
        # "model_": model_,
        "timesteps": pars_dict["m_per_timesteps"],
        "model_name": pars_dict["model_name"],
        "models_path": models_path,
        "model_file_name": model_file_name,
        "per_save_timesteps": pars_dict["per_save_timesteps"],
        "early_stopping_patience": pars_dict["early_stopping_patience"],
        "pre_model_file_name": None,
        "pre_step_name": "final"
    }
    predict_kwargs = {
        # "model_": model_,
        "models_path": models_path,
        "result_path": result_path,
        "model_file_name": model_file_name,
    }

    return train_kwargs, predict_kwargs


def begin_train(
        models_root_path,
        pars_dict,
        train_file_name,
        model_file_name,
        pre_model_path=None):
    models_path = f"{models_root_path}/{model_file_name}"
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    pars_dict_path = f"{models_path}/pars_dict.pickle"
    with open(pars_dict_path, 'wb') as handle:
        pickle.dump(pars_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log_path = f"{models_path}/output.log"
    print(f'out put save to {log_path}')

    if pre_model_path is None:
        os.system(
            f"nohup  {config.ENV_PYTHON_PATH} {config.PROJECT_PATH}/{train_file_name} -ap {pars_dict_path} -mfn {model_file_name} -p_t {pars_dict['predict_train']}>> {log_path} 2>&1 &")
    else:
        os.system(
            f"nohup  {config.ENV_PYTHON_PATH} {config.PROJECT_PATH}/{train_file_name} -ap {pars_dict_path} -mfn {model_file_name} -p_t {pars_dict['predict_train']} -pmp {pre_model_path}>> {log_path} 2>&1 &")
    print(f'Begin train of {model_file_name}')
    return model_file_name


# 实验_时间 / 参数组合 / 达尔文组合
def begin_trains(models_root_path, pars_dict, train_file_name):
    # log_name = f"{args.model_name}_{tic}_n{args.n_parallel_task}m{args.m_per_timesteps}M{args.M_tot_timesteps}_{datetime.datetime.now().strftime('%Y-%m%d-%H%M')}"
    for i in range(pars_dict["n_parallel_task"]):
        model_file_name = f"{i}"
        begin_train(
            models_root_path,
            pars_dict,
            train_file_name,
            model_file_name)
    return model_file_name


def cal_citerions(
        profit_train,
        profit_backtest,
        ration_train,
        ration_backtest):
    '''
    Criterion1 = train_daily_ratio * backtest_daily_ratio - (train_daily_ratio - backtest_daily_ratio) **2
    Criterion2 = train_times_ratio * backtest_times_ratio - (train_times_ratio - backtest_times_ratio) **2
    '''
    criterion_scaling = 1000000
    profits = profit_backtest * profit_train * criterion_scaling
    criterion_1 = (profit_train * profit_backtest -
                   (profit_train - profit_backtest) ** 2) * criterion_scaling
    criterion_2 = (ration_train * ration_backtest -
                   (ration_train - ration_backtest) ** 2) * criterion_scaling

    return criterion_1, criterion_2, profits


# main functions 4 monitor
# res dir: Darwin_name / pars_name / model_name
def record_info(df, Darwin_name, pars_name, model_name, step_model):
    path = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}"
    os.chdir(f"{path}/{Darwin_name}/{pars_name}/{model_name}")

    info_Darwin = extract_Darwin(Darwin_name, config.DARWIN_NAMES, 1)
    info_pars = extract_pars(pars_name, config.PARS_NAMES, 1)
    time_str = re.search(r'2021-([\S]+)', Darwin_name).group(1)
    if len(df) > 0:
        df.model_name = df.model_name.astype(str)
        if len(df[(df.info_Darwin == f"{info_Darwin}") & (df.info_pars == f"{info_pars}") & (
                df.time == time_str) & (df.model_name == model_name) & (df.step_model == step_model)]) > 0:
            return None
    # 4 backtest info
    trade_cache_name = f"trade_cache-backtest-{step_model}.csv"
    # trade_cache_file = f"{model_name}/{trade_cache_name}"
    trade_cache = pd.read_csv(trade_cache_name, index_col=0)
    profit_info = cal_profit(trade_cache)
    ratio = profit_info['total_ratio']
    ratio_range = profit_info['ratio_range']
    ratio_num = profit_info['ratio_num']
    trade_num = profit_info['trade_num']

    profit_backtest = ratio_range
    ratio_backtest = ratio
    ration_backtest = ratio_num
    traden_backtest = trade_num

    # 4 train info
    trade_cache_name = f"trade_cache-train-{step_model}.csv"
    trade_cache = pd.read_csv(trade_cache_name, index_col=0)
    profit_info = cal_profit(trade_cache)
    ratio = profit_info['total_ratio']
    ratio_range = profit_info['ratio_range']
    ratio_num = profit_info['ratio_num']
    trade_num = profit_info['trade_num']

    profit_train = ratio_range
    ratio_train = ratio
    ration_train = ratio_num
    traden_train = trade_num

    criterion_1, criterion_2, profits = cal_citerions(
        profit_train, profit_backtest, ration_train, ration_backtest)

    filter_flag = (criterion_1 > 0 and criterion_2 >
                   0 and ratio_backtest > 0 and ratio_train > 0)

    info = pd.DataFrame({
        'info_Darwin': [f"{info_Darwin}"],
        'info_pars': [f"{info_pars}"],
        'model_name': [model_name],
        'step_model': [step_model],

        'filter': [str(filter_flag)],
        'ratio_backtest': [ratio_backtest],
        'ratio_train': [ratio_train],
        'criterion_daily': [criterion_1],
        'criterion_times': [criterion_2],
        'profits': [profits],

        'profit_backtest': [profit_backtest],
        'ration_backtest': [ration_backtest],
        'traden_backtest': [traden_backtest],

        'profit_train': [profit_train],
        'ration_train': [ration_train],
        'traden_train': [traden_train],

        'time': [time_str],
    })

    return info


def MaxMinNormalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def monitor_record(Darwin_name, pars_name):
    path = f"{config.PROJECT_PATH}/{config.RESULTS_DIR}"
    os.chdir(f"{path}")
    # os.chdir(f"{path}/{Darwin_name}/{pars_name}")

    monitor_df_name = 'monitor_table'
    result_df = init_csv(f"{path}/{Darwin_name}", monitor_df_name)
    res_path = f"{path}/{Darwin_name}/{pars_name}"
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        return 0, pd.DataFrame()

    model_names = os.listdir(res_path)
    for model_name in model_names:
        step_models = os.listdir(
            f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{Darwin_name}/{pars_name}/{model_name}")
        for step_model in step_models:
            step_model = step_model[:-4]
            trade_cache_name_1 = f"trade_cache-backtest-{step_model}.csv"
            trade_cache_name_2 = f"trade_cache-train-{step_model}.csv"

            if os.path.exists(f"{path}/{Darwin_name}/{pars_name}/{model_name}/{trade_cache_name_1}") and os.path.exists(
                    f"{path}/{Darwin_name}/{pars_name}/{model_name}/{trade_cache_name_2}"):
                pass
            else:
                continue

            info = record_info(
                result_df,
                Darwin_name,
                pars_name,
                model_name,
                step_model)
            if info is None:
                print(
                    f"{Darwin_name}/{pars_name}/{model_name}/{step_model} already exists, ignore")
                continue
            result_df = pd.concat([result_df, info], axis=0)
    if len(result_df) < 1:
        return 0, result_df
    result_df = result_df.reset_index(drop=True)
    print(f"save {monitor_df_name}")
    result_df.to_csv(f"{path}/{Darwin_name}/{monitor_df_name}.csv")

    N_df = result_df[(result_df['filter']==True) | (result_df['filter']=='True')].copy()
    N_df = N_df.reset_index(drop=True)
    N = len(N_df)
    if N == 0:
        return 0, pd.DataFrame()
    sum_times_ratio = sum(N_df.ration_backtest) + sum(N_df.ration_train)
    sum_daily_ratio = sum(N_df.profit_backtest) + sum(N_df.profit_train)
    sum_criterion_1 = sum(N_df.criterion_daily)
    sum_criterion_2 = sum(N_df.criterion_times)
    sample_weight = sum_times_ratio / sum_daily_ratio * \
        (N_df.criterion_daily.values / sum_criterion_1 + N_df.criterion_times.values / sum_criterion_2)
    nor_sample_weight = MaxMinNormalization(sample_weight)
    N_df['sampling_weight'] = nor_sample_weight
    return N, N_df


def find_final(result_root_path, dones_list):
    model_names = os.listdir(result_root_path)

    for model_file_name in model_names:
        if model_file_name not in dones_list:
            result_path = f"{result_root_path}/{model_file_name}"
            res_files = os.listdir(result_path)

            if 'plot-train-final.jpg' in res_files:
                return model_file_name
    return False


def cal_res_finals(result_root_path):
    length = 0
    model_names = os.listdir(result_root_path)

    for model_file_name in model_names:
        result_path = f"{result_root_path}/{model_file_name}"
        if os.path.isdir(result_path):
            res_files = os.listdir(result_path)
            if 'plot-backtest-final.jpg' in res_files:
                length += 1
    return length


def random_choice(data, prob):
    res = 0
    try:
        res = random.choices(data, prob)[0]
    except BaseException:
        pass
    return res


def find_dir_len(result_root_path):
    length = 0
    res_names = os.listdir(result_root_path)
    for res_name in res_names:
        if os.path.isdir(f"{result_root_path}/{res_name}"):
            length += 1
    return length

# def filter_X_df(N_df,X):
#
