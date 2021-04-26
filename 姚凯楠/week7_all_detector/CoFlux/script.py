from src.CoFlux.Measure import CorrelationMeasurement
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
from src.CoFlux.predict_realization import holt_winter_predict
from src.CoFlux.predict_realization import wavelet_predict
from src.CoFlux.predict_realization import diff_predict
from src.CoFlux.predict_realization import *

import warnings

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    file_path = '../../file/pollution.csv'
    df = pd.read_csv(file_path)
    df = df.drop(['wnd_dir', 'date'], axis=1)
    df = df[:1500]
    df = (df - df.min()) / (df.max() - df.min())
    columns = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow']
    result_ground_truth = []
    result_holt_predict = [[] * 64] * 5
    result_tsd_median_predict = [[] * 4] * 5
    result_tsd_predict = [[] * 4] * 5
    result_diff_predict = [[] * 2] * 5
    result_his_predict = [[] * 4] * 5
    result_his_median_predict = [[] * 4] * 5
    result_wavelet_predict = [[] * 4] * 5
    for i in range(1):
        for j in range(i + 1, len(columns)):
            # for j in range(2, 3):

            cm = CorrelationMeasurement(0.63)

            # tsd median 4
            for x in range(1, 5):
                afx_set, afy_set = tsd_median_predict(df=df, column_name_x=columns[i],
                                                      column_name_y=columns[j], tsd_length=x * 24 * 7)
                result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                print('tsd median', result_cm)
                result_tsd_median_predict[j - 2].append(result_cm)
            # tsd 4
            for x in range(1, 5):
                afx_set, afy_set = tsd_predict(df=df, column_name_x=columns[i],
                                               column_name_y=columns[j], tsd_length=x * 24 * 7)
                result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                print('tsd median', result_cm)
                result_tsd_predict[j - 2].append(result_cm)

            # diff 2
            for x in range(1, 3):
                leng = 24
                if x == 2:
                    leng = 24 * 7
                afx_set, afy_set = diff_predict(df=df, column_name_x=columns[i], column_name_y=columns[j],
                                                diff_length=leng)
                result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                print('diff', result_cm)
                result_diff_predict[j - 2].append(result_cm)

            # his average 4
            for x in range(1, 5):
                afx_set, afy_set = his_average_predict(df=df, column_name_x=columns[i], column_name_y=columns[j],
                                                       average_length=x * 24 * 7)
                result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                print('his_average', result_cm)
                result_his_predict[j - 2].append(result_cm)

            # his median 4
            for x in range(1, 5):
                afx_set, afy_set = his_median_predict(df=df, column_name_x=columns[i], column_name_y=columns[j],
                                                      median_length=x * 24 * 7)
                result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                print('his_average', result_cm)
                result_his_median_predict[j - 2].append(result_cm)

            # holt winter 64
            for x in range(2, 10, 2):
                for y in range(2, 10, 2):
                    for k in range(2, 10, 2):
                        afx_set, afy_set = holt_winter_predict(df=df, test_length=24, column_name_x=columns[i],
                                                               column_name_y=columns[j], smoothing_level=x / 10,
                                                               smoothing_trend=y / 10, smoothing_seasonal=k / 10)
                        result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                        print('holtwinters', x / 10, y / 10, k / 10, result_cm)
                        result_holt_predict[j - 2].append(result_cm)
            # wavelet 4
            for x in range(1, 5):
                afx_set, afy_set = wavelet_predict(df=df, column_name_x=columns[i], column_name_y=columns[j],
                                                   test_length=(2 * x - 1) * 24)
                result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                print('his_average', result_cm)
                result_wavelet_predict[j - 2].append(result_cm)
    # afx_set, afy_set = holt_winter_predict(df=df, test_length=24, column_name_x='pollution',
    #                                        column_name_y='wnd_spd')
    # afx_set, afy_set = wavelet_predict(df=df, test_length=24, column_name_x='pollution', column_name_y='snow')
    # afx_set, afy_set = diff_predict(df=df, column_name_x='pollution', column_name_y='dew')
    #
    # cm = CorrelationMeasurement(0.7)
    # print(cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set))

    # result_wolt_winters = [(1, 0), (0, 0), (-1, 0), (-1, 1), (0, 0)]
    # result_wavelet = [(-1, 0), (1, 0), (-1, 0), (-1, 0), (-1, 0)]
    # result_diff = [(1, 1), (1, 1), (1, 1), (1, -1), (1, -1)]
    # result_his_aver = [(1, 1), (1, 1), (1, 1), (1, -1), (0, 0)]

    ground_true = [(1, 1), (1, 1), (-1, 0), (-1, 1), (0, 0)]

    function_column = np.zeros([5, 86])

    # 将对应的结果可用的detector存储在二维数组中，之后可以持久化或者存储在db中去使用

    for i in range(5):
        for j in range(len(result_holt_predict[i])):
            if result_holt_predict[i][j] == ground_true[i]:
                function_column[i][j] = 1
        for j in range(len(result_tsd_median_predict[i])):
            if result_tsd_median_predict[i][j] == ground_true[i]:
                function_column[i][64 + j] = 1
        for j in range(len(result_tsd_predict[i])):
            if result_tsd_predict[i][j] == ground_true[i]:
                function_column[i][68 + j] = 1
        for j in range(len(result_diff_predict[i])):
            if result_diff_predict[i][j] == ground_true[i]:
                function_column[i][72 + j] = 1
        for j in range(len(result_his_median_predict[i])):
            if result_his_median_predict[i][j] == ground_true[i]:
                function_column[i][74 + j] = 1
        for j in range(len(result_his_predict[i])):
            if result_his_predict[i][j] == ground_true[i]:
                function_column[i][78 + j] = 1
        for j in range(len(result_wavelet_predict[i])):
            if result_wavelet_predict[i][j] == ground_true[i]:
                function_column[i][82 + j] = 1

    print(function_column)
