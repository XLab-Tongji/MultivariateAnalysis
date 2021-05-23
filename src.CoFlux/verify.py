import numpy as np
import pandas as pd
from src.CoFlux.predict_realization import *
from src.CoFlux.Measure import CorrelationMeasurement
import time

t1 = time.time()
x = 0
listA = list()
listB = list()
listC = list()
for i in range(800):
    listA.append(np.sin(x * np.pi))
    listB.append(np.sin(x * np.pi - 0.5 * np.pi))
    x += 0.1

df = pd.DataFrame(columns={"listA", "listB"})
df["listA"] = listA
df["listB"] = listB
cm = CorrelationMeasurement(0)

afx_set, afy_set = get_result(df=df, column_name_x='listA', column_name_y='listB', period=20, oneDayLength=20)
print("start correlation")
result_cm = cm.correlation_measurement(afx_set=afx_set, afy_set=afy_set, max_dis=20)
print('cm', result_cm)
t2 = time.time()
print("Used Time: {}s".format(t2-t1))

# 问题1
# 我把period也作为参数提出到get_result()中了，不过感觉它在大多数情况下与oneDayLength应该是一致的？
# 除非几天一个周期或者一天包含几个周期才会不一致吧，这一点需要和你确认一下。

# 问题2：
# period=20， oneDayLength=20，20是当前的周期值，但是会报指针错误（这个bug需要修一下）

# 问题3：
# 测了几组数据，现在的cm结果不准确，问题应该出在get_result函数中
# 也许合理设置period与oneDayLength后可以得到更准确的cm，但是因为问题2中bug的存在，我无法测试。
# 下一步需要逐渐调试代码，直至verify.py的测试用例正确

# 其他：
# self.coTHR 默认值修改为0了，并且返回ccV，确保有结果能返回。
# 一些写死的常量24改为了变量period