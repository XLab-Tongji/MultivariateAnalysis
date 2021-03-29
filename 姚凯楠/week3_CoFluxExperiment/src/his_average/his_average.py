
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing

# 导入测试包
import sys

sys.path.append('../')
from src.util import indicators

train_len = 7 * 24 * 3
test_len = 24
offset = 0
column_name = 'OT'
train_set_size = train_len

file_path = '../../file/ETT-small/ETTh1.csv'
df = pd.read_csv(file_path)
train = df[offset:train_set_size + offset]
test = df[train_set_size + offset:train_set_size + test_len + offset]  # 测试数据

his_ava=df[column_name].rolling(window=24*7).mean()

y_hat_avg = test.copy()  # 拷贝测试数据
y_hat_avg['moving_avg_forecast'] = his_ava[train_set_size + offset:train_set_size + test_len + offset]
plt.figure(figsize=(16,8))
plt.plot(train[column_name], label='Train')
plt.plot(test[column_name], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()


print('mape:', indicators.mape(test[column_name], y_hat_avg['moving_avg_forecast']))
print('rmse:', indicators.rmse(test[column_name], y_hat_avg['moving_avg_forecast']))