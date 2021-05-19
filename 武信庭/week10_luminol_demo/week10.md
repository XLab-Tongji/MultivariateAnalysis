# 武信庭-第十周-Luminol_demo

## 本周进度

阅读luminol库代码与文档说明，应用库中的API完成demo，并总结归纳API功能

### 1. luminol

#### AnomalyDetector

异常检测接口

```python
from luminol.anomaly_detector import AnomalyDetector
ts1 = {0: 0, 1: 0.5, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}
detector = AnomalyDetector(ts1, score_threshold=1.5)
```

参数：

* `baseline_time_series`: 基线时间序列
* `score_threshold`: 超过这个值的异常分数将被识别为异常（可以覆盖 score_percentile_threshold）
* `score_precentile_threshold`：超过这个百分位数的异常分数将被识别为异常
* `algorithm_name(string)`: 使用特定的算法来计算异常得分
* `algorithm_params(dict)`：由algorithm_name指定的算法的附加参数
* `refine_algorithm_name(string)`：使用特定的算法来计算每个异常时期内的严重性时间戳
* `refine_algorithm_params(dict)`：由refine_algorithm_params指定的算法的附加参数

方法：

- `get_all_scores()`: 返回一个异常得分时间序列
* `get_anomalies()`: 返回一个异常对象的列表



#### Correlator

相关性检测

```python
from luminol.correlator import Correlator
for a in anomalies:
    time_period = a.get_time_window()
    my_correlator = Correlator(ts1, ts2, time_period)
    if my_correlator.is_correlated(threshold=0.8):
        print("ts2和ts1在时间窗口 (%d, %d) 上相关" % time_period)
```

参数：

* `time_series_a`: 第一个时间序列
* `time_series_b`: 第二个时间序列
* `time_period(tuple)`: 一个时间段，用于关联这两个时间序列。
* `use_anomaly_score(bool)`：用时间序列的异常分数将来计算相关系数，而不是时间序列的原始数据。
* `algorithm_name`：用特定的算法来计算相关系数。
* `algorithm_params`：由algorithm_name指定的算法的任何额外参数。

方法：

- `get_correlation_result()`：返回一个[CorrelationResult](#modules) 对象。
* `is_correlated(threshold=0.7)`：如果系数高于传入的阈值，返回一个[CorrelationResult](#modules)对象。否则，返回false。



#### Demo

将AnomalyDetector与Correlator等接口应用于简单数据集上试验效果并熟悉接口操作，具体代码见同路径下luminol_demo.py