import os
import time as tm
from finrl.config import config
import sys
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
import warnings
import pickle
import random
from test_packages import *
from Darwin_packages import *
warnings.filterwarnings("ignore")
sys.path.append("../FinRL-multi")

os.chdir(f"{config.PROJECT_PATH}")

py_file_name = "Darwin_script.py"
train_file_name = "Darwin_train.py"


# ############### Argparse ###############
def get_parse():
    parser = argparse.ArgumentParser(description='')
    # 4 env
    parser.add_argument(
        "-model",
        "--model_name",
        help="model name",
        type=str,
        default="dqn")
    parser.add_argument(
        "-r",
        "--r_negative_minus",
        help="r_negative_minus",
        type=int,
        default=20)
    parser.add_argument(
        "-rr",
        "--rr_positive_mul",
        help="rr_positive_mul",
        type=int,
        default=20)
    parser.add_argument("-hr", "--hr_mul", help="hr_mul", type=int, default=1)
    parser.add_argument(
        "-ce",
        "--ce_positive_mul",
        help="ce_positive_mul",
        type=int,
        default=4)
    parser.add_argument(
        "-tr",
        "--tr_bm_coef",
        help="tr_bm_coef",
        type=int,
        default=3)
    # 4 train
    parser.add_argument(
        "-pst",
        "--per_save_timesteps",
        help="per_save_timesteps",
        type=int,
        default=2)
    parser.add_argument(
        "-esp",
        "--early_stopping_patience",
        help="early_stopping_patience",
        type=int,
        default=None)
    # 4 predict
    parser.add_argument(
        "-p_t",
        "--predict_train",
        help="whether predict train",
        type=bool,
        default=True)
    # 4 parallel train
    parser.add_argument(
        "-n",
        "--n_parallel_task",
        help="n_parallel_task",
        type=int,
        default=5)
    parser.add_argument(
        "-m",
        "--m_per_timesteps",
        help="m_per_timesteps",
        type=int,
        default=10)
    parser.add_argument(
        "-M",
        "--M_tot_timesteps",
        help="M_tot_timesteps",
        type=int,
        default=100)
    parser.add_argument(
        "--tic",
        help="tic label",
        type=str,
        default="AAPL")

    args = parser.parse_args()
    return args


# if __name__=="__main__":
args = get_parse()
pars_dict = vars(args)
# ###########################################################
env_kwargs = {
    "r_negative_minus": pars_dict["r_negative_minus"],
    "rr_positive_mul": pars_dict["rr_positive_mul"],
    "hr_mul": pars_dict["hr_mul"],
    "ce_positive_mul": pars_dict["ce_positive_mul"],
    "tr_bm_coef": pars_dict["tr_bm_coef"]
}
pars_kwargs = merge_dict(env_kwargs, {})

Darwin_info = Darwin_str(pars_dict)
external_info = pars_str(pars_kwargs)

pars_dict['Darwin_info'] = Darwin_info
pars_dict['external_info'] = external_info
# ###########################################################
# root_path : /{Darwin_info}/{external_info}
# /{model_file_name}/{steps~~}
_, models_root_path = get_models_path(Darwin_info, external_info, "")
_, result_root_path = get_result_path(Darwin_info, external_info, "")
last_model_name = begin_trains(models_root_path, pars_dict, train_file_name)
# ###########################################################

# Darwin_info = "dqn_AAPL_n5m10M100_2021-0616-0445"
# external_info = "r20rr10hr1ce4tr3"
# last_model_name = "4"
# _, models_root_path = get_models_path(Darwin_info, external_info, "")
# _, result_root_path = get_result_path(Darwin_info, external_info, "")
#
# pars_dict['Darwin_info'] = Darwin_info
# pars_dict['external_info'] = external_info

acc_timesteps = 0
dones_list = []
sleep_time = 60
n = pars_dict["n_parallel_task"]
m = pars_dict["m_per_timesteps"]
M = pars_dict["M_tot_timesteps"]
flag_M = False

while True:

    # update criterions record
    N, N_df = monitor_record(Darwin_info, external_info)

    # if final -> new one
    model_file_name = find_final(result_root_path, dones_list)
    while model_file_name:
        dones_list.append(model_file_name)
        acc_timesteps += m
        flag_M = (acc_timesteps >= M)

        if flag_M:
            print('end training!')
            # Wait for traning end
            #
            len_models = find_dir_len(models_root_path)
            len_res_finals = cal_res_finals(result_root_path)
            while len_models != len_res_finals:
                pass
            N, N_df = monitor_record(Darwin_info, external_info)
            # output top X paths and criterions
            X = min(10, N)
            break
        elif (N == 0) or (random.random() < 1/(1+N/2)):
            # Start a new random task
            # last_model_name_int = int(last_model_name)
            last_model_name_int = find_dir_len(models_root_path)
            new_model_name = str(last_model_name_int + 1)
            pre_model_path = None
        else:
            random_index = random_choice(
                N_df.index.values, N_df.sampling_weight.values)
            df_temp = N_df.loc[random_index]
            # create new name
            # 1(1s)-2(2s)-3(4s)-5(final)-6   |  ?s/final |
            # len(os.listdir()) - 1
            new_model_name = f"{df_temp.model_name}_{df_temp.step_model}-{find_dir_len(models_root_path)}"
            pre_model_path = f"{models_root_path}/{df_temp.model_name}/{df_temp.step_model}"

        last_model_name = begin_train(
            models_root_path,
            pars_dict,
            train_file_name,
            new_model_name,
            pre_model_path)

        model_file_name = find_final(result_root_path, dones_list)

    if flag_M:
        print('end Darwin!')
        break
    time.sleep(sleep_time)
