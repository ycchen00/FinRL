import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from finrl.config import config
import random

class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df,state_space=24):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """

        if self.use_technical_indicator == True:
            # add technical indicators using stockstats
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence == True:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature == True:
            df = self.add_user_defined_feature(df,state_space)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=['tic','date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic','date',indicator]],on=['tic','date'],how='left')
        df = df.sort_values(by=['date','tic'])
        return df

    def add_user_defined_feature(self,df,state_space=24, span=15, thred=0.02, std=0.2, k_d=2):
        """
        add user defined features
        :param df: (df) pandas dataframe
        :return: (df) pandas dataframe
        """

        def generate_trend(df, span=15):
            s = df.close
            sma = s.rolling(window=span, min_periods=span).mean()[:span]
            rest = s[span:]
            temp = pd.concat([sma, rest]).ewm(span=span, adjust=False).mean()

            res = temp
            res[:span] = s[:span]
            return res

        def generate_trend_feature(df_stock):
            t_pct = np.array(df_stock.trend.pct_change(1))
            # 计算trend近5天变化率
            df_stock.loc[:, "t_5"] = np.around(
                np.append([0] * 6, t_pct[1:-5]), k_d)
            df_stock.loc[:, "t_4"] = np.around(
                np.append([0] * 5, t_pct[1:-4]), k_d)
            df_stock.loc[:, "t_3"] = np.around(
                np.append([0] * 4, t_pct[1:-3]), k_d)
            df_stock.loc[:, "t_2"] = np.around(
                np.append([0] * 3, t_pct[1:-2]), k_d)
            df_stock.loc[:, "t_1"] = np.around(
                np.append([0] * 2, t_pct[1:-1]), k_d)
            df_stock.loc[:, "t_0"] = np.around(np.append([0] * 1, t_pct[1:]), k_d)
            return df_stock

        def generate_violatrend_feature(df_stock):
            trend = df_stock.trend
            violation = df_stock.close

            vt_pct = (violation - trend) / trend

            df_stock.loc[:, "vt_5"] = np.around(
                np.append([0] * 6, vt_pct[1:-5]), k_d)
            df_stock.loc[:, "vt_4"] = np.around(
                np.append([0] * 5, vt_pct[1:-4]), k_d)
            df_stock.loc[:, "vt_3"] = np.around(
                np.append([0] * 4, vt_pct[1:-3]), k_d)
            df_stock.loc[:, "vt_2"] = np.around(
                np.append([0] * 3, vt_pct[1:-2]), k_d)
            df_stock.loc[:, "vt_1"] = np.around(
                np.append([0] * 2, vt_pct[1:-1]), k_d)
            df_stock.loc[:, "vt_0"] = np.around(
                np.append([0] * 1, vt_pct[1:]), k_d)
            return df_stock

        def generate_reward_feature(df_stock):
            # hold 时reward特征
            df_stock.loc[:, "rh_1"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_2"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_3"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_4"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_5"] = np.zeros(len(df_stock))

            # buy 后 reward特征
            df_stock.loc[:, "rb_1"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_2"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_3"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_4"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_5"] = np.zeros(len(df_stock))

            return df_stock

        def nor(udh):
            abs_udh = list(map(abs, udh))
            temp = sum(abs_udh)
            u, d, h = [x / temp for x in abs_udh]
            return u, d, h

        def prob_area(u, u_std=0.3, d_std=0.3, h_std=0.3):
            udh = np.zeros((u.shape[0], 3))
            d = np.zeros((u.shape))
            h = np.zeros((u.shape))
            for i in range(u.shape[0]):
                val = u[i]
                if val == 0.8:
                    noise = random.gauss(0, u_std)
                    randomp = random.random()
                    u[i] = val + noise
                    d[i] = (1 - val) / 3 - noise * randomp
                    h[i] = 2 * (1 - val) / 3 - noise * (1 - randomp)
                    if u[i] < 0 or d[i] < 0 or h[i] < 0:
                        u[i], d[i], h[i] = nor([u[i], d[i], h[i]])
                elif val == 0.2:
                    noise = random.gauss(0, d_std)
                    randomp = random.random()
                    u[i] = val + noise
                    d[i] = 2 * (1 - val) / 3 - noise * randomp
                    h[i] = (1 - val) / 3 - noise * (1 - randomp)
                    if u[i] < 0 or d[i] < 0 or h[i] < 0:
                        u[i], d[i], h[i] = nor([u[i], d[i], h[i]])
                else:  # ==0.5
                    randompp = random.sample([1, 2], 1)[0]
                    noise = random.gauss(0, h_std)
                    randomp = random.random()
                    u[i] = val + noise
                    d[i] = randompp * (1 - val) / 3 - noise * randomp
                    h[i] = (3 - randompp) * (1 - val) / 3 - noise * (1 - randomp)
                    if u[i] < 0 or d[i] < 0 or h[i] < 0:
                        u[i], d[i], h[i] = nor([u[i], d[i], h[i]])

            udh[:, 0] = u[:]
            udh[:, 1] = d[:]
            udh[:, 2] = h[:]
            return udh

        def prob_areas(df_stock, thre=0.02, std=0.2, k_d=2,span=15):
            probs = []
            c_pct = np.array(df_stock.close.pct_change(1))

            # 计算close近5天变化率
            #         df_stock.loc[:, "c_5"] = np.around(np.append([0] * 5, c_pct[:-5]), k_d)
            #         df_stock.loc[:, "c_4"] = np.around(np.append([0] * 4, c_pct[:-4]), k_d)
            #         df_stock.loc[:, "c_3"] = np.around(np.append([0] * 3, c_pct[:-3]), k_d)
            #         df_stock.loc[:, "c_2"] = np.around(np.append([0] * 2, c_pct[:-2]), k_d)
            #         df_stock.loc[:, "c_1"] = np.around(np.append([0] * 1, c_pct[:-1]), k_d)
            #         df_stock.loc[:, "c_0"] = np.around(c_pct[:], k_d)

            c = c_pct[1:]
            u = np.ones(c.shape) * 0.5

            u[c > thre] = 0.8
            u[c < -thre] = 0.2

            udh = prob_area(u, std, std, std)
            probs.append(udh)

            # 模拟计算概率
            df_stock.loc[:, "u"] = np.around(
                np.append([0] * 1, probs[0][:, 0]), k_d)
            df_stock.loc[:, "d"] = np.around(
                np.append([0] * 1, probs[0][:, 1]), k_d)
            df_stock.loc[:, "h"] = np.around(
                np.append([0] * 1, probs[0][:, 2]), k_d)

            #         df_stock.drop(["c_0"], axis=1, inplace=True)

            # hold 时reward特征
            df_stock.loc[:, "rh_1"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_2"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_3"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_4"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rh_5"] = np.zeros(len(df_stock))

            # buy 后 reward特征
            df_stock.loc[:, "rb_1"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_2"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_3"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_4"] = np.zeros(len(df_stock))
            df_stock.loc[:, "rb_5"] = np.zeros(len(df_stock))

            skip_index = span + 6
            return df_stock[skip_index:]  # [6:]


        def add_user_defined_feature_24(df,span=15, thred=0.02, std=0.2, k_d=2):
            df = df.sort_values(by=['tic', 'date'])
            df['trend'] = generate_trend(df)

            stock = Sdf.retype(df.copy())
            unique_ticker = stock.tic.unique()
            skip_index = span + 6

            _df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_data = stock[stock.tic == unique_ticker[i]]
                    df_temp = temp_data[["tic", "close", "trend"]].copy()
                    df_temp = generate_violatrend_feature(df_temp)
                    df_temp = generate_trend_feature(df_temp)
                    temp_indicator = prob_areas(df_temp, thre=thred, std=std, k_d=k_d, span=span)
                    # temp_indicator = temp_indicator.iloc[skip_index:]
                    temp_indicator.reset_index("date", inplace=True)
                    #             temp_indicator.loc[:, "day_stamp"] = np.arange(len(temp_indicator))

                    _df = _df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            _df = _df.sort_values(by=['date', 'tic'])
            # _df.index = _df.date.factorize()[0]
            order = ["date", "tic", "close", "trend", "u", "d", "h",
                     "t_5", "t_4", "t_3", "t_2", "t_1",
                     "vt_5", "vt_4", "vt_3", "vt_2", "vt_1",
                     "rh_5", "rh_4", "rh_3", "rh_2", "rh_1",
                     "rb_5", "rb_4", "rb_3", "rb_2", "rb_1"]

            return _df[order]


        def add_user_defined_feature_37(df):
            df = df.sort_values(by=['tic', 'date'])
            df['trend'] = generate_trend(df)

            stock = Sdf.retype(df.copy())
            unique_ticker = stock.tic.unique()
            skip_index = span + 6

            _df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_data = stock[stock.tic == unique_ticker[i]]
                    df_temp = temp_data.copy()
                    df_temp = generate_violatrend_feature(df_temp)
                    df_temp = generate_trend_feature(df_temp)
                    # temp_indicator = prob_areas(df_temp, thre=thred, std=std, k_d=k_d, span=span)
                    temp_indicator = generate_reward_feature(df_temp)
                    # temp_indicator = temp_indicator.iloc[skip_index:]
                    round_dict = {}
                    for feature in config.TECHNICAL_USERDEFINED_LIST_36:
                        round_dict[feature]=2
                    temp_indicator = temp_indicator.round(round_dict)

                    temp_indicator.reset_index("date", inplace=True)
                    #             temp_indicator.loc[:, "day_stamp"] = np.arange(len(temp_indicator))

                    _df = _df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            _df = _df.sort_values(by=['date', 'tic'])

            order = ["date", "tic", "close", "trend"]+config.TECHNICAL_USERDEFINED_LIST_36
            return _df[order]

        if state_space == 24:
            _df=add_user_defined_feature_24(df)
        elif state_space == 37:
            _df = add_user_defined_feature_37(df)
        return _df


    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        return None

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index
