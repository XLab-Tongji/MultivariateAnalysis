import numpy as np
import pywt
from pandas import Series
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from src.util import indicators
import warnings

warnings.filterwarnings('ignore')


def holt_winter_predict(df, column_name, period=24, smoothing_level=0.6, smoothing_trend=0.6,
                        smoothing_seasonal=0.6):
    result = []
    train_init_length = period * 7
    # 分段预测 每段预测后24个值 训练长度为24*7
    for i in range(int((len(df) - train_init_length) / period)):
        train_length = train_init_length + i * 24
        train_data = df[i * 24:train_length]
        test_data = df[train_length:train_length + 24]
        y_hat_avg = test_data.copy()  # 拷贝测试数据
        # 模型和预测
        fit_model = ExponentialSmoothing(
            np.asarray(train_data[column_name]),
            seasonal_periods=period, trend='add', seasonal='add')
        fit_data = fit_model.fit(smoothing_level=smoothing_level, smoothing_seasonal=smoothing_seasonal,
                                 smoothing_trend=smoothing_trend)
        y_hat_avg['Holt_Winter'] = fit_data.forecast(len(test_data))
        # 得到波动值 预测减原始值
        l1 = list(y_hat_avg['Holt_Winter'])
        l2 = list(test_data[column_name])
        for index in range(len(l1)):
            result.append(l1[index] - l2[index])

    j = int((len(df) - train_init_length) / period) - 1
    start = 24 * (j + 7 + 1)
    end = len(df)
    last_train_data = df[start - train_init_length:start]
    last_test_data = df[start:end]
    last_copy = last_test_data.copy()
    fit_model = ExponentialSmoothing(np.asarray(last_train_data[column_name]),
                                     seasonal_periods=period, trend='add', seasonal='add')
    fit_data = fit_model.fit(smoothing_level=smoothing_level, smoothing_seasonal=smoothing_seasonal,
                             smoothing_trend=smoothing_trend)
    last_copy['Holt_Winter'] = fit_data.forecast(len(last_test_data))
    l1 = list(last_copy['Holt_Winter'])
    l2 = list(last_test_data[column_name])
    for index in range(len(l1)):
        result.append(l1[index] - l2[index])
    return result


# 差分 选择24 或者24*7
def diff_predict(df, column_name, length=24):
    result = []
    df_column = df[column_name]
    # 预测值是上周或者昨天的值
    # 原始值当天的值 则在本例中是向后偏移一天或者一周的值
    s1 = list(df_column[length:])

    # 预测减原始
    # 原始值长度比较短，所以用预测值数组长度来循环
    for i in range(len(s1)):
        result.append(df_column[i] - s1[i])
    return result


# 历史平均 选择宽度为1，2，3，4*7*24
def his_average_predict(df, column_name, length=24 * 7):
    result = []
    df_x = df[column_name]

    # 预测值 选择窗口为window
    s1 = df_x.rolling(window=length).mean()
    # 预测减原始
    for i in range(len(s1)):
        if i < length - 1:
            continue
        result.append(s1[i] - df_x[i])
    return result


# 历史中值 选择宽度为1，2，3，4*7*24
def his_median_predict(df, column_name, length=24 * 7):
    result = []
    df_x = df[column_name]
    # 预测值 选择窗口为window
    s1 = df_x.rolling(window=length).median()
    # 预测减原始
    for i in range(len(s1)):
        if i < length - 1:
            continue
        result.append(s1[i] - df_x[i])
    return result


def tsd_predict(df, column_name, length=24 * 7, period=24):
    result_x = seasonal_decompose(df[column_name], model='additive', period=period)
    result = result_x.resid.rolling(window=length).mean()
    result = result[length + int(period / 2) - 1:-int(period / 2)]
    return result


def tsd_median_predict(df, column_name, length=24 * 7, period=24):
    result_x = seasonal_decompose(df[column_name], model='additive', period=period)
    result = result_x.resid.rolling(window=length).median()
    result = result[length + int(period / 2) - 1:-int(period / 2)]
    return result


def wavelet(data):
    db4 = pywt.Wavelet('db4')

    coeffs = pywt.wavedec(data, db4)
    coeffs[0] *= 0

    meta = pywt.waverec(coeffs, db4)
    return meta


def wavelet_predict(df, column_name, length=24 * 1):
    # 设置500为阈值 再低预测效果不好
    # 如果length小于500 则设置成500
    length = max(500, length)
    # 有了一个最大值之后，确保能和df长度对齐
    length = min(len(df), length)
    result = []
    df_x = df[column_name]
    number = int(len(df_x) / length)
    for i in range(number):
        df_temp = list(df_x[i * length:(i + 1) * length])
        pre_result = wavelet(df_temp)
        for j in range(len(df_temp)):
            result.append(pre_result[j] - df_temp[j])
    # 补齐可能会缺失的部分
    for i in range(1):
        df_temp=list(df_x[number*length:])
        pre_result=wavelet(df_temp)
        for j in range(len(df_temp)):
            result.append(pre_result[j]-df_temp[j])
    return result


def get_af_set(df, column_name, period=24, oneDayLength=24):
    result = []
    # 由于tsd最后period/2个数字是nan 所以其他的都要删除最后period/2对齐
    # 两种diff 1 day 1 week
    print("now diff")
    minLength = len(df)
    # 尾对齐长度
    offsetLength = int(period / 2)
    result.append(diff_predict(df=df, column_name=column_name, length=oneDayLength)[:-offsetLength])
    result.append(diff_predict(df=df, column_name=column_name, length=oneDayLength * 7)[:-offsetLength])
    # 64种holt winters
    print("now holt winters")
    for i in range(2, 10, 2):
        for j in range(2, 10, 2):
            for k in range(2, 10, 2):
                result.append(holt_winter_predict(df=df, column_name=column_name, period=period,
                                                  smoothing_level=i / 10, smoothing_trend=j / 10,
                                                  smoothing_seasonal=k / 10)[:-offsetLength])
    print("now his and tsd")
    for i in range(1, 5):
        result.append(his_average_predict(df=df, column_name=column_name, length=oneDayLength * 7 * i)[:-offsetLength])
        result.append(his_median_predict(df=df, column_name=column_name, length=oneDayLength * 7 * i)[:-offsetLength])
        # # 两种tsd不用尾对齐
        result.append(tsd_predict(df=df, column_name=column_name, length=oneDayLength * 7 * i, period=period))
        result.append(tsd_median_predict(df=df, column_name=column_name, length=oneDayLength * 7 * i, period=period))

    print("now wavelet")
    for i in range(1, 8, 2):
        result.append(wavelet_predict(df=df, column_name=column_name, length=oneDayLength*i)[:-offsetLength])

    for item in result:
        minLength = min(minLength, len(item))
    print('当前有效长度是:',minLength)
    af_set = []
    for item in result:
        af_set.append(item[len(item) - minLength:])
    return af_set


def get_result(df, column_name_x, column_name_y, oneDayLength=24):
    afx_set = get_af_set(df=df, column_name=column_name_x, oneDayLength=oneDayLength)
    afy_set = get_af_set(df=df, column_name=column_name_y, oneDayLength=oneDayLength)
    return afx_set, afy_set


if __name__ == "__main__":
    print("hello world")
