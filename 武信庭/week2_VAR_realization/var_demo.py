import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv("./AirQualityUCI.csv", parse_dates=[['Date', 'Time']])

# 查看列数据类型
# print(df.dtypes)

# 预处理数据
# 修改日期类型
df['Date_Time'] = pd.to_datetime(df.Date_Time, format='%Y/%m/%d %H:%M:%S')
data = df.drop(['Date_Time'], axis=1)

# 建立日期索引
data.index = df.Date_Time

# 缺失值处理
cols = data.columns
data = data.replace(-200, np.nan)
data = data.fillna(method='ffill')
print("初始数据：",data)

# 检查平稳性
johan_test_temp = data.drop(['CO(GT)'], axis=1)
stationarity_result = coint_johansen(johan_test_temp, -1, 1).eig
print("平稳性检测：", stationarity_result)

# 创建训练集与验证集
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

# 模型训练
model = VAR(endog=train)
model_fit = model.fit()

# 在验证集上进行预测
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

# 将预测结果转为df格式
pred = pd.DataFrame(index=range(0,len(prediction)), columns=[cols])
for j in range(0, 13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]
pred.index = data.index[int(0.8*(len(data))):]

# pred[:] = pred[:].astype(int)
# valid[:] = valid[:].astype(int)
# print(pred.dtypes)
# print(valid.dtypes)
# print(pred.columns[1])

# 检查rmse
for i in range(0, 13):
    print('rmse value for', cols[i], 'is : ', sqrt(mean_squared_error(pred.iloc[i], valid.iloc[i])))

# 在完整数据集上拟合
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)
print(data)

# 可视化
plt.figure()
plt.plot(valid.loc[:, ["PT08.S1(CO)"]], label='GroundTruth')
plt.plot(pred.loc[:, ["PT08.S1(CO)"]], label='Prediction')
plt.legend()
plt.show()