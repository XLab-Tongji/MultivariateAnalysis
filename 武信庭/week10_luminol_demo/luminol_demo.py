from luminol.anomaly_detector import AnomalyDetector
from luminol.correlator import Correlator

ts1 = {0: 0, 1: 0.5, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}
ts2 = {0: 0, 1: 0.5, 2: 1, 3: 0.5, 4: 1, 5: 0, 6: 1, 7: 1, 8: 1}

detector = AnomalyDetector(ts1, score_threshold=1.5)  # 设定阈值
anomalies = detector.get_anomalies()
score = detector.get_all_scores()

print("时间序列点与其对应的异常打分为：")
for timestamp, value in score.iteritems():
    print(timestamp, value)

try:
    print("异常时间窗口为", anomalies[0].get_time_window())  # 输出异常时间窗口
except IndexError:
    print("没有检测出异常值!")

for a in anomalies:
    time_period = a.get_time_window()
    my_correlator = Correlator(ts1, ts2, time_period)
    if my_correlator.is_correlated(threshold=0.8):
        print("ts2和ts1在时间窗口 (%d, %d) 上相关" % time_period)
