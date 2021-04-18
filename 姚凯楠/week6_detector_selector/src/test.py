import warnings

from src.CoFlux.Measure import CorrelationMeasurement
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
from src.CoFlux.predict_realization import holt_winter_predict
from src.CoFlux.predict_realization import wavelet_predict
from src.CoFlux.predict_realization import diff_predict
from src.CoFlux.predict_realization import his_average_predict

if __name__ == '__main__':
    column = [[0, 0, 1, 1],
              [0, 0, 1, 1],
              [1, 1, 0, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 1]]
    warnings.filterwarnings('ignore')
    file_path = '../../file/pollution.csv'
    df = pd.read_csv(file_path)
    df = df.drop(['wnd_dir', 'date'], axis=1)
    df = df[:1000]
    df = (df - df.min()) / (df.max() - df.min())
    columns = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow']
    cm = CorrelationMeasurement(0.63)

    for i in range(len(column)):
        for j in range(len(column[i])):
            if column[i][j] == 1:
                if j == 0:
                    afx_set, afy_set = holt_winter_predict(df=df, test_length=24, column_name_x=columns[i],
                                                           column_name_y=columns[j])
                    result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                    print('holt winters', result_cm)
                elif j == 1:
                    afx_set, afy_set = wavelet_predict(df=df, test_length=24, column_name_x=columns[i],
                                                       column_name_y=columns[j])
                    result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                    print('wavelet', result_cm)
                elif j == 2:
                    afx_set, afy_set = diff_predict(df=df, column_name_x=columns[i], column_name_y=columns[j])
                    result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                    print('diff', result_cm)
                elif j == 3:
                    afx_set, afy_set = his_average_predict(df=df, column_name_x=columns[i], column_name_y=columns[j])
                    result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
                    print('his_average', result_cm)
                break
