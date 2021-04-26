# week7_all_detector

1. 实现了所有的detector的
2. detector
   1. tsd median 时间序列分解 然后将残差的中位数作为波动值
   2. tsd 时间序列分解 然后将残差作为波动值
   3. diff 差分，跨度为1或者7天
   4. history average 历史平均
   5. history median 历史中值
   6. holt winters 将参数分别设置为0.2 0.4 0.6 0.8 实际产生64个detector
   7. wavelet 做小波分解预测 每次预测步长为5