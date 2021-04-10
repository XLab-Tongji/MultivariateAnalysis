from src.CoFlux.Measure import CorrelationMeasurement
import pandas as pd
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
from src.CoFlux.predict_realization import holt_winter_predict
from src.CoFlux.predict_realization import wavelet_predict

from src.util import indicators

if __name__ == '__main__':
    file_path = '../../file/pollution.csv'
    df = pd.read_csv(file_path)
    df = df.drop(['wnd_dir', 'date'], axis=1)
    df = df[:500]

    # for i in range(1, 10):
    #     for j in range(1, 10):
    #         for k in range(1, 10):
    #             traindata=df[:400]
    #             testdata=df[400:424]
    #             y_hat_avg = testdata.copy()  # 拷贝测试数据
    #             fit_model = ExponentialSmoothing(
    #                 np.asarray(traindata['pollution']),
    #                 seasonal_periods=24,
    #                 trend='add',
    #                 seasonal='add')
    #             fit_data = fit_model.fit(
    #                 smoothing_level=i/10,
    #                 smoothing_trend=j/10,
    #                 smoothing_seasonal=k/10
    #             )
    #             y_hat_avg['Holt_Winter'] = fit_data.forecast(len(testdata))
    #             print('mape:',i,j,k, indicators.mape(testdata['pollution'], y_hat_avg['Holt_Winter']))

    # df = (df - df.min()) / (df.max() - df.min())
    afx_set, afy_set = holt_winter_predict(df=df, test_length=24, column_name_x='pollution',column_name_y='snow')
    # afx_set, afy_set = wavelet_predict(df=df, test_length=24, column_name_x='pollution', column_name_y='snow')
    cm = CorrelationMeasurement()
    cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
