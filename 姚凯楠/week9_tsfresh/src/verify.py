import numpy as np
import pandas as pd
from src.CoFlux.predict_realization import *
from src.CoFlux.Measure import CorrelationMeasurement

startValueA = 0
startValueB = -30
startValueC = -180
listA = list()
listB = list()
listC = list()
for i in range(1500):
    listA.append(np.sin(startValueA * np.pi / 180))
    listB.append(np.sin(startValueB * np.pi / 180))
    listC.append(np.sin(startValueC * np.pi / 180))
    startValueA += 15
    startValueB += 15
    startValueC += 15

df = pd.DataFrame(columns={"listA", "listB", "listC"})
df["listA"] = listA
df["listB"] = listB
df["listC"] = listC
cm = CorrelationMeasurement(0.63)

afx_set, afy_set = get_result(df=df, column_name_x='listA', column_name_y='listB')
print("start correlation")
result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set)
print('cm', result_cm)
