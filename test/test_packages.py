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
from pprint import pprint
import sys
sys.path.append("../FinRL-multi")

def data_split2(df,ratio=0.7):
    thre = int(len(df)*ratio)
    train = df.iloc[:thre].copy()
    trade = df.iloc[thre:].copy()
    train.index = train.date.factorize()[0]
    trade.index = trade.date.factorize()[0]
    return train, trade


def save_final_asset(trade_cache):
    asset = trade_cache['asset']
    filter_asset = asset[asset > 0]
    return sum(filter_asset), len(filter_asset)


def train_save_models(
        agent,
        model_file_name,
        model_,
        model_name,
        timesteps,
        models_path,
        per_save_timesteps=None,
        early_stopping_patience=None,
        pre_model_file_name=None,
        pre_step_name='final'):
    print(f'\n**Begin train of {models_path}/{model_file_name}**')

    if pre_model_file_name is not None:
        env_train = agent.env
        pre_models_path=f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}/{pre_model_file_name}"
        print(f"Load pre_train model {pre_model_file_name}")
        model_ = model_.load(f"pre_models_path/{pre_step_name}",env_train)


    trained_ = agent.train_model(
        model=model_,
        tb_log_name=model_name,
        total_timesteps=timesteps * 10000,
        save_path=models_path,
        per_save_timesteps=per_save_timesteps,
        early_stopping_patience=early_stopping_patience)

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    trained_.save(f"{models_path}/final")
    print(f'**Success train of {model_file_name}**\n')
    return trained_


def predict_save_model(
        e_trade_gym,
        external_info,
        model_file_name,
        model_,
        result_path,
        models_path,
        step_name="final"):
    print(f"Predicting {model_file_name}:{step_name}/{external_info}")
    trained_model = model_.load(f"{models_path}/{step_name}")

    # backtest dataset
    trade_cache, trade_state_memory, empty_state_memory = DRLAgent.DRL_prediction(
        model=trained_model, environment=e_trade_gym)
    final_asset, trade_num = save_final_asset(trade_cache)
    asset_txt = f"{final_asset}_{trade_num}"
    f = open(f"{result_path}/{external_info}-{step_name}-{asset_txt}.txt", "a")

    trade_cache.to_csv(
        f"{result_path}/trade_cache-{external_info}-{step_name}.csv")
    trade_state_memory.to_csv(
        f"{result_path}/trade_state_memory-{external_info}-{step_name}.csv")
    empty_state_memory.to_csv(
        f"{result_path}/empty_state_memory-{external_info}-{step_name}.csv")

    trade = e_trade_gym.df
    tics = trade.tic.unique().tolist()

    trade_plot(
        trade,
        trade_cache,
        tics,
        f"{model_file_name}-{external_info}-{step_name}")
    # plt.show()
    plt.savefig(f"{result_path}/plot-{external_info}-{step_name}.jpg")
    print(f'Success')
    return True


def predict_save_models(
        models_path,
        e_trade_gym,
        model_file_name,
        model_,
        result_path,
        predict_train=False,
        e_train_gym=None,
        exact_model_name=None):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    predict_kwargs = {
        "model_": model_,
        "models_path": models_path,
        "result_path": result_path,
        "model_file_name": model_file_name,
    }
    if exact_model_name is not None:
        step_name = exact_model_name
        predict_save_model(
            step_name=step_name,
            e_trade_gym=e_trade_gym,
            external_info="backtest",
            **predict_kwargs)
        # train dataset
        if predict_train and e_train_gym is not None:
            predict_save_model(
                step_name=step_name,
                e_trade_gym=e_train_gym,
                external_info="train",
                **predict_kwargs)
    else:
        print('\n*****Predicting (every step)*****')
        model_names = os.listdir(models_path)
        model_names.sort(reverse=True)
        for step_name in model_names:
            if not step_name.endswith('.zip'):
                continue
            step_name = step_name.strip('.zip')

            predict_save_model(
                step_name=step_name,
                e_trade_gym=e_trade_gym,
                external_info="backtest",
                **predict_kwargs)
            # train dataset
            if predict_train and e_train_gym is not None:
                predict_save_model(
                    step_name=step_name,
                    e_trade_gym=e_train_gym,
                    external_info="train",
                    **predict_kwargs)
    return True


def get_processed_full(tic, state_space, flag_pro=True):
    if flag_pro:
        processed_full = pd.read_csv(
            f"./{config.DATA_SAVE_DIR}/data_{tic}_user{state_space - 1}.csv",
            index_col=0)
        # processed_full = pd.read_csv("./datasets/data_30S_user18.csv",index_col=0)
    else:
        if state_space == 24:
            processed_full = get_processed_full_24(tic)
        elif state_space == 37:
            processed_full = get_processed_full_37(tic)

    return processed_full


def get_processed_full_37(tic,state_space=37):
    raw_csv_name = "df_orig_raw.csv"
    prob_csv_name = "rf_df_model_result.csv"

    prob_df = read_prob_df(prob_csv_name, tic)
    raw_df = read_raw_df(raw_csv_name, tic)

    df = pd.merge(prob_df, raw_df, on=['date', 'tic'], how='inner')

    df.sort_values(["date", "tic"], ignore_index=True)

    fe = FeatureEngineer(
        use_technical_indicator=False,
        user_defined_feature=True
    )

    processed = fe.preprocess_data(df, state_space)

    processed_full = processed.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(0)

    processed_full.sort_values(["date", "tic"], ignore_index=True)
    processed_full.index = processed_full.date.factorize()[0]

    return processed_full

def get_processed_full_24(tic):
    df = pd.read_csv(
        f"./{config.DATA_SAVE_DIR}/data_{tic}.csv",
        index_col=0)

    df.sort_values(["date", "tic"], ignore_index=True)

    fe = FeatureEngineer(
        use_technical_indicator=False,
        user_defined_feature=True
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(
            processed["date"].min(),
            processed["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(
        combination, columns=[
            "date", "tic"]).merge(
        processed, on=[
            "date", "tic"], how="left")
    processed_full = processed_full[processed_full["date"].isin(
        processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(0)

    processed_full.sort_values(["date", "tic"], ignore_index=True)
    processed_full.index = processed_full.date.factorize()[0]

    return processed_full


def read_prob_df(prob_csv_name,tic=None):
    df = pd.read_csv(f"./{config.DATA_SAVE_DIR}/{prob_csv_name}",encoding='utf-8')
    rename_dict = {'rand_time': 'date',
                   'ts_code': 'tic',
                   'prob_Y_-3_1d_df':'d1_3',
                  'prob_Y_-2_1d_df' :'d1_2',
                  'prob_Y_-1_1d_df' :'d1_1',
                  'prob_Y_0_1d_df'  :'h1',
                  'prob_Y_1_1d_df'  :'u1_1',
                  'prob_Y_2_1d_df'  :'u1_2',
                  'prob_Y_3_1d_df'  :'u1_3',
                  'pred_Y_1_df'     :'p1',
                  'prob_Y_-3_5d_df' :'d5_3',
                  'prob_Y_-2_5d_df' :'d5_2',
                  'prob_Y_-1_5d_df' :'d5_1',
                  'prob_Y_0_5d_df'  :'h5',
                  'prob_Y_1_5d_df'  :'u5_1',
                  'prob_Y_2_5d_df'  :'u5_2',
                  'prob_Y_3_5d_df'  :'u5_3',
                  'pred_Y_5_df': 'p5'
    }
    df.rename(columns=rename_dict, inplace=True)
    columns = [
        'date',
        'tic',
        'd1_3',
        'd1_2',
        'd1_1',
        'h1',
        'u1_1',
        'u1_2',
        'u1_3',
        'p1',
        'd5_3',
        'd5_2',
        'd5_1',
        'h5',
        'u5_1',
        'u5_2',
        'u5_3',
        'p5'
    ]
    return df[df.tic==tic][columns]


def read_raw_df(raw_csv_name,tic=None):
    df = pd.read_csv(f"./{config.DATA_SAVE_DIR}/{raw_csv_name}",encoding='utf-8')
    df.rename(columns={'trade_time': 'date', 'ts_code': 'tic'}, inplace=True)
    columns = ['date', 'tic', 'close']
    if tic is None:
        return df[columns]
    return df[df.tic==tic][columns]