import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt  # python 小波变换的包
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA

# 取数据
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
show=df[offset:train_set_size + test_len + offset]
train = df[offset:train_set_size + offset]
test = df[train_set_size + offset:train_set_size + test_len + offset]  # 测试数据
y_hat_avg = test.copy()  # 拷贝测试数据

show_list=np.array(show[column_name])
index_list = np.array(train[column_name])  # 最后10个数据排除用来做预测
date_list1 = np.array(train['date'])

index_for_predict = np.array(test[column_name])  # 预测的真实值序列
date_list2 = np.array(test['date'])

# 分解
A2, D2, D1 = pywt.wavedec(index_list, 'db4', mode='sym', level=2)  # 分解得到第4层低频部分系数和全部4层高频部分系数
coeff = [A2, D2, D1]

a=1

# 对每层小波系数求解模型系数
order_A2 = sm.tsa.arma_order_select_ic(A2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q

print( order_A2,order_D2,order_D1 )  #各个模型阶次

# 对每层小波系数构建ARMA模型
model_A2 =  ARMA(A2,order=order_A2)   # 建立模型
model_D2 =  ARMA(D2,order=order_D2)
model_D1 =  ARMA(D1,order=order_D1)

results_A2 = model_A2.fit()
results_D2 = model_D2.fit()
results_D1 = model_D1.fit()

# 画出每层的拟合曲线
plt.figure(figsize=(10,15))
plt.subplot(3, 1, 1)
plt.plot(A2, 'blue')
plt.plot(results_A2.fittedvalues,'red')
plt.title('model_A2')

plt.subplot(3, 1, 2)
plt.plot(D2, 'blue')
plt.plot(results_D2.fittedvalues,'red')
plt.title('model_D2')

plt.subplot(3, 1, 3)
plt.plot(D1, 'blue')
plt.plot(results_D1.fittedvalues,'red')
plt.title('model_D1')


a=1


A2_all,D2_all,D1_all = pywt.wavedec(np.array(df['OT']),'db4',mode='sym',level=2) # 对所有序列分解
delta = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)] # 求出差值，则delta序列对应的为每层小波系数ARMA模型需要预测的步数

print( delta)
# 预测小波系数 包括in-sample的和 out-sample的需要预测的小波系数
pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(A2)+delta[0])
pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[1])
pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[2])

# 重构
coeff_new = [pA2,pD2,pD1]
denoised_index = pywt.waverec(coeff_new,'db4')

# 画出重构后的原序列预测图
plt.figure(figsize=(15,5))
plt.plot(train.index, train['OT'], label='Train')
plt.plot(test.index, test['OT'], label='Test')
plt.plot(denoised_index[:train_set_size + test_len + offset],'red')

plt.show()

print('mape:', indicators.mape(test[column_name], denoised_index[train_set_size + offset:train_set_size + test_len + offset]))
print('rmse:', indicators.rmse(test[column_name], denoised_index[train_set_size + offset:train_set_size + test_len + offset]))
