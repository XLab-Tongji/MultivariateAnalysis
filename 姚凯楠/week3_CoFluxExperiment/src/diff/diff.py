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
y_hat_avg = test.copy()  # 拷贝测试数据

dd = np.asarray(train['OT'])
y_hat = test.copy()
y_hat['naive'] = dd[len(dd) - test_len:len(dd)]
plt.figure(figsize=(12, 8))
plt.plot(train.index, train['OT'], label='Train')
plt.plot(test.index, test['OT'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

print('mape:', indicators.mape(test[column_name], y_hat['naive']))
print('rmse:', indicators.rmse(test[column_name], y_hat['naive']))
