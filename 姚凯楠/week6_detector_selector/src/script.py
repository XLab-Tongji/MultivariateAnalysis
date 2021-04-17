from src.CoFlux.Measure import CorrelationMeasurement
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
from src.CoFlux.predict_realization import holt_winter_predict
from src.CoFlux.predict_realization import wavelet_predict
from src.CoFlux.predict_realization import diff_predict
from src.CoFlux.predict_realization import his_average_predict

import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    file_path = '../../file/pollution.csv'
    df = pd.read_csv(file_path)
    df = df.drop(['wnd_dir', 'date'], axis=1)
    df = df[:1000]
    df = (df - df.min()) / (df.max() - df.min())
    columns = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow']
    result_ground_truth = []
    result_wolt_winters = []
    result_wavelet = []
    result_diff = []
    result_his_aver = []
    for i in range(1):
        for j in range(i + 1, len(columns)):
            # print(columns[i], columns[j])
            # df_ttt = df[[columns[i], columns[j]]]
            # df_ttt.plot()
            # plt.show()

            cm = CorrelationMeasurement(0.63)

            afx_set, afy_set = holt_winter_predict(df=df, test_length=24, column_name_x=columns[i],
                                                   column_name_y=columns[j])
            result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
            print('holtwinters', result_cm)
            result_wolt_winters.append(result_cm)

            afx_set, afy_set = wavelet_predict(df=df, test_length=24, column_name_x=columns[i],
                                               column_name_y=columns[j])
            result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
            print('wavelet', result_cm)
            result_wavelet.append(result_cm)

            afx_set, afy_set = diff_predict(df=df, column_name_x=columns[i], column_name_y=columns[j])
            result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
            print('diff', result_cm)
            result_diff.append(result_cm)

            afx_set, afy_set = his_average_predict(df=df, column_name_x=columns[i], column_name_y=columns[j])
            result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
            print('his_average', result_cm)
            result_his_aver.append(result_cm)

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

    function_column = np.zeros([5,4])

    # 将对应的结果可用的detector存储在二维数组中，之后可以持久化或者存储在db中去使用
    # 1 表示holt winter
    # 2 表示 wavelet
    # 3 表示 diff
    # 4 表示 his_average

    for i in range(len(result_wolt_winters)):
        if result_wolt_winters[i] == ground_true[i]:
            print('when measure ', columns[i + 1], 'can use holt winters')
            function_column[i][0] = 1
        if result_wavelet[i] == ground_true[i]:
            print('when measure ', columns[i + 1], 'can use wavelet')
            function_column[i][1] = 1
        if result_diff[i] == ground_true[i]:
            print('when measure ', columns[i + 1], 'can use diff')
            function_column[i][2] = 1
        if result_his_aver[i] == ground_true[i]:
            print('when measure ', columns[i + 1], 'can use his average')
            function_column[i][3] = 1

    print(function_column)
