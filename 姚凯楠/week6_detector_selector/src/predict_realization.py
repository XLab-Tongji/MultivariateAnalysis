import numpy as np
import pywt
from pandas import Series
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot as plt

from src.util import indicators
import warnings

warnings.filterwarnings('ignore')


def holt_winter_predict(df, test_length, column_name_x, column_name_y, smoothing_level=0.6, smoothing_trend=0.6,
                        smoothing_seasonal=0.6, trend='add', seasonal='add'):
    afx_set = []
    afy_set = []
    df = df
    test_length = test_length
    seasonal_periods = test_length
    train_init_length = test_length * 7
    column_name_x = column_name_x
    column_name_y = column_name_y
    smoothing_level = smoothing_level
    smoothing_trend = smoothing_trend
    smoothing_seasonal = smoothing_seasonal
    trend = trend
    seasonal = seasonal
    content_x = []

    for i in range(int((len(df) - 7 * 24) / 24)):
        train_length = train_init_length + i * 24
        # result
        train_data = df[:train_length]
        test_data = df[train_length:train_length + test_length]
        y_hat_avg = test_data.copy()  # 拷贝测试数据
        fit_model = ExponentialSmoothing(
            np.asarray(train_data[column_name_x]),
            seasonal_periods=seasonal_periods)
        fit_data = fit_model.fit()
        y_hat_avg['Holt_Winter'] = fit_data.forecast(len(test_data))

        l1 = list(y_hat_avg['Holt_Winter'])
        l2 = list(test_data[column_name_x])
        for i in range(len(test_data)):
            content_x.append(l1[i] - l2[i])

    content_y = []
    for i in range(int((len(df) - 7 * 24) / 24)):
        train_length = train_init_length + i * 24
        # result
        train_data = df[:train_length]
        test_data = df[train_length:train_length + test_length]
        y_hat_avg = test_data.copy()  # 拷贝测试数据
        fit_model = ExponentialSmoothing(
            np.asarray(train_data[column_name_y]),
            seasonal_periods=seasonal_periods)
        fit_data = fit_model.fit()
        y_hat_avg['Holt_Winter'] = fit_data.forecast(len(test_data))
        l1 = list(y_hat_avg['Holt_Winter'])
        l2 = list(test_data[column_name_y])
        for i in range(len(test_data)):
            content_y.append(l1[i] - l2[i])

    number = int(len(content_x) / (test_length * 7))
    for i in range(number):
        tempList = []
        tempList2 = []
        for j in range(i * test_length * 7, i * test_length * 7 + test_length * 7):
            tempList.append(content_x[j])
            tempList2.append(content_y[j])
        afx_set.append(tempList)
        afy_set.append(tempList2)
    return afx_set, afy_set


def wavelet(df):
    index_list = np.array(df)
    A2, D2, D1 = pywt.wavedec(index_list, 'db4', mode='sym', level=2)
    coeff = [A2, D2, D1]
    # 模型系数
    order_A2 = sm.tsa.arma_order_select_ic(A2, ic='aic')['aic_min_order']  # AIC准则求解模型阶数p,q
    order_D2 = sm.tsa.arma_order_select_ic(D2, ic='aic')['aic_min_order']  # AIC准则求解模型阶数p,q
    order_D1 = sm.tsa.arma_order_select_ic(D1, ic='aic')['aic_min_order']  # AIC准则求解模型阶数p,q
    # 建立模型
    model_A2 = ARMA(A2, order=order_A2)  # 建立模型
    model_D2 = ARMA(D2, order=order_D2)
    model_D1 = ARMA(D1, order=order_D1)

    results_A2 = model_A2.fit()
    results_D2 = model_D2.fit()
    results_D1 = model_D1.fit()

    A2_all, D2_all, D1_all = pywt.wavedec(np.array(df), 'db4', mode='sym', level=2)  # 对所有序列分解
    delta = [len(A2_all) - len(A2), len(D2_all) - len(D2),
             len(D1_all) - len(D1)]  # 求出差值，则delta序列对应的为每层小波系数ARMA模型需要预测的步数

    pA2 = model_A2.predict(params=results_A2.params, start=1, end=len(A2) + delta[0])
    pD2 = model_D2.predict(params=results_D2.params, start=1, end=len(D2) + delta[1])
    pD1 = model_D1.predict(params=results_D1.params, start=1, end=len(D1) + delta[2])

    coeff_new = [pA2, pD2, pD1]
    denoised_index = pywt.waverec(coeff_new, 'db4')

    # plt.plot(df.index,df,label='pollution')
    # plt.plot(denoised_index,color='red')
    # plt.show()

    return Series(denoised_index)


def wavelet_predict(df, test_length, column_name_x, column_name_y):
    afx_set = []
    afy_set = []
    test_length = test_length
    column_name_x = column_name_x
    column_name_y = column_name_y
    df = df
    df_x = df[column_name_x]
    df_y = df[column_name_y]
    scaler = StandardScaler()

    # predict x
    s1 = wavelet(df=df_x)
    s2 = wavelet(df=df_y)
    number = int(len(df) / test_length)
    for i in range(number):
        tempList = []
        tempList2 = []
        for j in range(i * test_length, i * test_length + test_length):
            tempList.append(s1[i] - df_x[i])
            tempList2.append(s2[i] - df_y[i])

        afx_set.append(tempList)
        afy_set.append(tempList2)

    return afx_set, afy_set


def diff_predict(df, column_name_x, column_name_y, diff_length=24):
    afx_set = []
    afy_set = []
    df_x = df[column_name_x]
    df_y = df[column_name_y]
    s1 = list(df_x[24:])
    s2 = list(df_y[24:])
    scaler = StandardScaler()

    for i in range(len(s1)):
        s1[i] = s1[i] - df_x[i]
        s2[i] = s2[i] - df_y[i]
    number = int(len(s1) / (24 * 7))
    for i in range(number):
        tempList = []
        tempList2 = []
        for j in range(i * diff_length * 7, i * diff_length * 7 + diff_length * 7):
            tempList.append(s1[j] - df_x[j])
            tempList2.append(s2[j] - df_y[j])

        afx_set.append(tempList)
        afy_set.append(tempList2)
    return afx_set, afy_set


def his_average_predict(df, column_name_x, column_name_y, average_length=24):
    afx_set = []
    afy_set = []
    df_x = df[column_name_x]
    df_y = df[column_name_y]
    s1 = df[column_name_x].rolling(window=average_length).mean()
    s2 = df[column_name_y].rolling(window=average_length).mean()

    for i in range(len(s1)):
        s1[i] = s1[i] - df_x[i]
        s2[i] = s2[i] - df_y[i]

    number = int(len(s1) / (24 * 7))
    for i in range(1, number):
        tempList = []
        tempList2 = []
        for j in range(i * average_length * 7, i * average_length * 7 + average_length * 7):
            tempList.append(s1[j] - df_x[j])
            tempList2.append(s2[j] - df_y[j])

        afx_set.append(tempList)
        afy_set.append(tempList2)
    return afx_set, afy_set
