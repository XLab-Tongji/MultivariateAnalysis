# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


class TimeSeriesSplit():
    def __init__(self, series, EMA):
        '''
        时间序列分解算法，乘法模型，由于循环波动难以确认，受随机因素影响大，不予考虑
        series：时间序列
        EMA：移动平均项数，也是周期的时长
        '''
        self.buildModel(series, EMA)

    def predict(self, num):
        '''
        往后预测num个数，返回的是整个模型的信息
        num：预测个数
        '''
        result = []
        for i in range(num):
            # 季节因子
            S = self.seasFactors[(i + len(self.series)) % len(self.seasFactors)]
            # 长期趋势
            T = self.regression.predict(i + len(self.series))[0][0]
            result.append(T * S)
        info = {
            'predict': {'value': result, 'desc': '往后预测的%s个数' % num},
            'Ta': {'value': self.regression.coef_[0][0], 'desc': '长期趋势线性模型的系数'},
            'Tb': {'value': self.regression.intercept_[0], 'desc': '长期趋势线性模型的截距'},
            'seasonFactor': {'value': self.seasFactors, 'desc': '季节因子'},
        }
        return info

    def buildModel(self, series, EMA):
        '''
        建模，预测
        series：时间序列
        EMA：移动平均项数，也是周期的时长
        '''
        series = np.array(series).reshape(-1)
        # 移动平均数
        moveSeies = self.calMoveSeries(series, EMA)
        # 季节因子
        seasonFactors = self.calSeasonFactors(series, moveSeies, EMA)
        # 长期趋势建模
        regression = self.buildLongTrend(series)
        # 收尾，设置对象属性
        self.series = series
        self.seasFactors = seasonFactors
        self.regression = regression

    def calMoveSeries(self, series, EMA):
        '''
        计算移动平均数
        series：时间序列
        EMA：移动平均项数，也是周期的时长
        '''
        # 计算移动平均
        moveSeries = []
        for i in range(0, series.shape[0] - EMA + 1):
            moveSeries.append(series[i:i + EMA].mean())
        moveSeries = np.array(moveSeries).reshape(-1)
        # 如果项数为复数，则移动平均后数据索引无法对应原数据，要进行第2次项数为2的移动平均
        if EMA % 2 == 0:
            moveSeries2 = []
            for i in range(0, moveSeries.shape[0] - 2 + 1):
                moveSeries2.append(moveSeries[i:i + 2].mean())
            moveSeries = np.array(moveSeries2).reshape(-1)
        return moveSeries

    def calSeasonFactors(self, series, moveSeries, EMA):
        '''
        计算季节性因子
        series：时间序列
        moveSeries：移动平均数
        EMA：移动平均项数，也是周期的时长
        '''
        # 移动平均后的第一项索引对应原数据的索引
        startIndex = int((series.shape[0] - moveSeries.shape[0]) / 2)
        # 观测值除以移动平均值
        factors = []
        for i in range(len(moveSeries)):
            factors.append(series[startIndex + i] / moveSeries[i])
        # 去掉尾部多余部分
        rest = len(factors) % EMA
        factors = factors[:len(factors) - rest]
        factors = np.array(factors).reshape(-1, EMA)

        # 平均值可能不是1,调整
        seasonFactors = factors.mean(axis=0) / factors.mean()
        # 按季顺序进行索引调整
        seasonFactors = seasonFactors[startIndex:].reshape(-1).tolist() + seasonFactors[:startIndex].reshape(
            -1).tolist()
        seasonFactors = np.array(seasonFactors).reshape(-1)
        return seasonFactors

    def buildLongTrend(self, series):
        '''
        计算长期趋势
        series：时间序列
        '''
        # 建立线性模型
        reg = LinearRegression()
        # 季节索引从0开始
        index = np.array(range(series.shape[0])).reshape(-1, 1)
        reg.fit(index, series.reshape(-1, 1))
        return reg


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     # https://wenku.baidu.com/view/89933c24a417866fb94a8e0a.html?from=search
#     # https://blog.csdn.net/weixin_40159138/article/details/90603344
#     # 销售数据
#     data =np.array ([
#         3017.60, 3043.54, 2094.35, 2809.84,
#         3274.80, 3163.28, 2114.31, 3024.57,
#         3327.48, 3439.48, 3493.93, 3490.79,
#         3685.08, 3661.23, 2378.43, 3459.55,
#         3849.63, 3701.18, 2642.38, 3585.52,
#         4078.66, 3907.06, 2818.46, 4089.50,
#         4339.61, 4148.60, 2976.45, 4084.64,
#         4242.42, 3997.58, 2881.01, 4036.23,
#         4360.33, 4360.53, 3172.18, 4223.76,
#         4690.48, 4694.48, 3342.35, 4577.63,
#         4965.46, 5026.05, 3470.14, 4525.94,
#         5258.71, 5489.58, 3596.76, 3881.60
#     ])
#     data=data.reshape(-1,1)
#     # plt.plot(range(len(data)),data)
#     model = TimeSeriesSplit(data, 4)
#     # 往后预测4个数，也就是1年4个季度的数
#     print(model.predict(4))