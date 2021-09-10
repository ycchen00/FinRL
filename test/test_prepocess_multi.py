import os
import itertools
import pandas as pd
import numpy as np
from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer
import sys
sys.path.append("../FinRL-multi")

df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/data_AAPL.csv",index_col=0)
# df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/data_30S.csv",index_col=0)

# df = df.sort_values(by=['tic','date'])
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
processed_full.to_csv("./" + config.DATA_SAVE_DIR + "/data_AAPL_user23.csv")
print('succeed')