import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

# 读取数据
df = pd.read_csv("./pollution.csv")

# 预处理数据
# 修改日期类型
df['date'] = pd.to_datetime(df.date, format='%Y/%m/%d %H:%M:%S')

# 建立日期索引
data = df.drop(['date'], axis=1)
data.index = df.date
print("初始数据：", data)

# 创建训练集与验证集
nobs = 10
train = data[:-nobs]
valid = data[-nobs:]
print("train:", train.shape)
print("valid:", valid.shape)


test = data[:int(0.01 * (len(data)))]
# 检查相关性
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
test.plot(ax=ax1)  # series plot
pd.plotting.lag_plot(test)  # 1ag plot
plt. show()

# ADF检验
def adf(time_series):
    result = adfuller(time_series.values)
    print(' ADF statistic:%f' % result[0])
    print(' p-value:%f' % result[1])
    print(' Critical Values:')
    for key, value in result[4].items():
        print('\t%s:%.3f' % (key, value))


# print('Augmented Dickey-Fuller Test:dew Time Series')
# adf(train['dew'])
# print('Augmented Dickey-Fuller Test:temp Time Series')
# adf(train['temp'])
# print('Augmented Dickey-Fuller Test:press Time Series')
# adf(train['press'])
# print('Augmented Dickey-Fuller Test:wnd_dir Time Series')
# adf(train['wnd_dir'])
# print('Augmented Dickey-Fuller Test:wnd_spd Time Series')
# adf(train['wnd_spd'])
# print('Augmented Dickey-Fuller Test:snow Time Series')
# adf(train['snow'])
# print('Augmented Dickey-Fuller Test:rain Time Series')
# adf(train['rain'])
# print('Augmented Dickey-Fuller Test:pollution Time Series')
# adf(train['pollution'])

# 一阶差分
train_diff = train.diff().dropna()
print(train_diff)
# train_diff.plot(figsize=(10, 6))
# plt.show()

# # 检验平稳性
# print('Augmented Dickey-Fuller Test:dew Time Series')
# adf(train_diff['dew'])
# print('Augmented Dickey-Fuller Test:temp Time Series')
# adf(train_diff['temp'])
# print('Augmented Dickey-Fuller Test:press Time Series')
# adf(train_diff['press'])
# print('Augmented Dickey-Fuller Test:wnd_dir Time Series')
# adf(train_diff['wnd_dir'])
# print('Augmented Dickey-Fuller Test:wnd_spd Time Series')
# adf(train_diff['wnd_spd'])
# print('Augmented Dickey-Fuller Test:snow Time Series')
# adf(train_diff['snow'])
# print('Augmented Dickey-Fuller Test:rain Time Series')
# adf(train_diff['rain'])
# print('Augmented Dickey-Fuller Test:pollution Time Series')
# adf(train_diff['pollution'])

# print(grangercausalitytests(train_diff[['dew', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['temp', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))#
# print(grangercausalitytests(train_diff[['press', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))#
# print(grangercausalitytests(train_diff[['wnd_dir', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['wnd_spd', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))
# print(grangercausalitytests(train_diff[['snow', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))#
# print(grangercausalitytests(train_diff[['rain', 'pollution']],
#                             maxlag=15, addconst=True, verbose=True))#
#
# 模型初始化
model = VAR(endog=train_diff)

# 训练
model_fit = model.fit()
# model_fit.summary()

# 选择lag order
lag_order = model_fit.k_ar
print("lag order为：", lag_order)

input_data = train_diff.values[-lag_order:]
print("预测数据为：", input_data)

# 预测
pred = model_fit.forecast(y=input_data, steps=nobs)
pred = (pd.DataFrame(pred, index=valid.index, columns=valid.columns))
print(pred)

# 还原差分 de-differentiate
forecast = pred.copy()
columns = train.columns
for col in columns:
    forecast[col] = train[col].iloc[-1] + forecast[col].cumsum()
print(forecast)

# 图像输出
plt.figure(figsize=(12, 5))
plt.xlabel('date')

ax1 = valid.pollution.plot(color='blue', grid=True, label='Actual pollution')
ax2 = forecast.pollution.plot(color='red', grid=True, label='Predicted pollution')
ax1.legend(loc=1)
ax2.legend(loc=2)
plt.show()
