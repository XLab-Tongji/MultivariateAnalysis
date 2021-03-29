import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing

# 导入测试包
import sys

sys.path.append('../')
from src.util import indicators

# 导入数据
train_len = 0.015
file_path = '../../file/ETT-small/ETTh1.csv'
df = pd.read_csv(file_path)
train_set_size = int((len(df)) * 0.8 * train_len)
train = df[:train_set_size]
test = df[train_set_size:int(train_set_size)+24]  # 测试数据
y_hat_avg = test.copy()  # 拷贝测试数据

# 设置参数
column_name = 'OT'
seasonal_periods = 24
trend = 'add'
seasonal = 'add'
smoothing_level = 0.9
smoothing_trend = 0.07
smoothing_seasonal = 0.4
print('run holt winters: ')
print('column name:', column_name, ',', 'trend:', trend, ',', 'seasonal:', seasonal, ',', 'smooth_level:',
      smoothing_level, ',', 'smooth_trend:', smoothing_trend, ',', 'smooth_seasonal:', smoothing_seasonal)

# holt winters
fit1 = ExponentialSmoothing(np.asarray(train[column_name]), seasonal_periods=seasonal_periods, trend=trend,
                            seasonal=seasonal).fit(
    smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, smoothing_seasonal=smoothing_seasonal
)

# 可视化
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.plot(train[column_name], label='Train')
plt.plot(test[column_name], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

print('mape:', indicators.mape(test[column_name], y_hat_avg['Holt_Winter']))
print('rmse:', indicators.rmse(test[column_name], y_hat_avg['Holt_Winter']))
