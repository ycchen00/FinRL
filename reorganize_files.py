import os
import time as tm
from finrl.config import config
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append("../FinRL-multi")
os.chdir(f"{config.PROJECT_PATH}/")

models_path=f"{config.PROJECT_PATH}/{config.TRAINED_MODEL_DIR}"
result_path=f"{config.PROJECT_PATH}/{config.RESULTS_DIR}"

model_names = os.listdir(models_path)
os.chdir(models_path)

res_names = os.listdir(result_path)
os.chdir(result_path)

names = res_names
for name in names:
    if len(name)>10:
        if 'a2c' in name:
    # name = 'a2c_AAPL_multi_u23_t20w_r20rr20hr5ce2tr4_2021-0609-0519'
            os.system(f"mv {name} a2c/")
        elif 'ppo' in name:
            os.system(f"mv {name} ppo/")
        else:
            continue
        print(f"{name} moved!")
pass