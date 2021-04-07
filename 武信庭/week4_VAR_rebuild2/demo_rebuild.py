import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
from math import sqrt
from sklearn.metrics import mean_squared_error

# 读取数据
df = pd.read_csv("./pollution.csv")

# 查看列数据类型
print(df.dtypes)

# 预处理数据
# 修改日期类型
df['date'] = pd.to_datetime(df.date, format='%Y/%m/%d %H:%M:%S')

# 建立日期索引
data = df.drop(['date'], axis=1)
data.index = df.date

# # 缺失值处理
# data = data.replace(-200, np.nan)
# data = data.replace(-200.0, np.nan)
# data = data.fillna(method='ffill')
print("初始数据：", data)

# 检查平稳性
johan_test_temp = data.drop(['press'], axis=1)
stationarity_result = coint_johansen(johan_test_temp, -1, 1).eig
print("平稳性检测：", stationarity_result)

# # 创建训练集与验证集
# train = data[:int(0.8 * (len(data)))]
# valid = data[int(0.8 * (len(data))):]
# print("valid:" )
#
# # 模型训练
# model = VAR(endog=train)
# model_fit = model.fit()
# cols = data.columns
#
# # 在验证集上进行预测
# prediction = model_fit.forecast(model_fit.y, steps=len(valid))
# print(prediction)
#
# # 将预测结果转为df格式
# pred = pd.DataFrame(index=range(0, len(prediction)), columns=[cols])
# for j in range(0, 8):
#     for i in range(0, len(prediction)):
#         pred.iloc[i][j] = prediction[i][j]
# pred.index = data.index[int(0.8 * (len(data))):]
# print(pred)

train_len = int(0.8 * (len(data)))
left_len = int(0.2 * (len(data)))
# 创建训练集与验证集
train = data[:train_len]
valid = data[train_len:]
# print(valid.index[0])
print("init train df is: ", train)
print("valid length is: ", left_len)

for i in range(left_len):
    # 模型训练
    model = VAR(endog=train)
    model_fit = model.fit()
    cols = data.columns

    # 在验证集上进行一步预测
    prediction = model_fit.forecast(model_fit.y, steps=1)
    # print("prediction is: ", prediction)

    ind = valid.index[i]
    # 将预测结果转为df格式并加入训练集
    train.loc[ind] = prediction[0]
    # print(train)

print("final train is:", train)
pred = train.copy(deep=True)

# pred[:] = pred[:].astype(int)
# valid[:] = valid[:].astype(int)
# print(pred.dtypes)
# print(valid.dtypes)


# def mean_absolute_percentage_error(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 检查rmse与mape
for i in range(0, 8):
    print('rmse value for', cols[i], 'is : ', sqrt(mean_squared_error(pred.iloc[i], valid.iloc[i])))
    # print(' mape value for', cols[i], 'is : ', mean_absolute_percentage_error(valid.iloc[i], pred.iloc[i]))


# # 在完整数据集上拟合
# model = VAR(endog=data)
# model_fit = model.fit()
# yhat = model_fit.forecast(model_fit.y, steps=1)
# print(yhat)
# print(data)
#
# 可视化
plt.figure()
# plt.plot(valid.loc[:'2014-01-03 02:00:00', ["pollution"]], label='GroundTruth')
plt.plot(valid.loc[:, ["pollution"]], label='GroundTruth')
plt.plot(pred.loc['2014-01-01 00:00:00':, ["pollution"]], label='Prediction')
plt.legend()
plt.show()
