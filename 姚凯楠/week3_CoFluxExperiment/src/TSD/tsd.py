from src.Module.BuildModel import TimeSeriesSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

train_len = 7 * 24 * 3
test_len = 24
offset = 0
column_name = 'OT'
train_set_size = train_len

file_path = '../../file/ETT-small/ETTh1.csv'
df = pd.read_csv(file_path)
train = df[offset:train_set_size + offset]['OT'].values
test = df[train_set_size + offset:train_set_size + test_len + offset]['OT'].values  # 测试数据
y_hat_avg = test.copy()  # 拷贝测试数据

EMA = 24  # 周期长度，即12个月
model = TimeSeriesSplit(train, EMA)
# 预测
result = model.predict(len(test))
print('季节性因子', np.round(result['seasonFactor']['value'], 2))
print('长期趋势系数和截距', np.round(result['Ta']['value'], 2), np.round(result['Tb']['value'], 2))
print('预测值', np.round(result['predict']['value'], 2))

plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(len(result['predict']['value'])), result['predict']['value'], label="predict")
plt.plot(range(len(result['predict']['value'])), test, label="real")
plt.legend()
plt.title('时间序列分解法预测效果')
plt.show()
