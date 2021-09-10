import sys
sys.path.append("../FinRL-multi")

from test_packages import *
############################## Unchanged pars ##############################
# 4 data
flag_pro = False
tic = "000001.SZ"
# tic = "AAPL"
TECHNICAL_USERDEFINED_LIST = config.TECHNICAL_USERDEFINED_LIST_36
stock_dimension = 1
state_space = len(TECHNICAL_USERDEFINED_LIST) + 1
processed_full = get_processed_full(tic, state_space, flag_pro)

tics = processed_full.tic.unique().tolist()

train,trade = data_split2(processed_full,0.7)
print('Dataset info:')
print(f'trainset len: {len(train)}')
print(f'tradeset len: {len(trade)}\n')

pass