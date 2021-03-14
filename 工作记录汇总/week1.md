## 武信庭

### 多变量时间序列数据集

1. 空气污染数据集: https://github.com/PatientEz/CNN-BiLSTM-Attention-Time-Series-Prediction_Keras
2. 飞机引擎寿命数据集: https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM/tree/master/Dataset
3. **房屋能源使用情况**: https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/tree/master/data

NASA Prognostics Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan



### 多变量时间序列预测的传统方法

[Normalizing Kalman Filtersfor Multivariate Time Series Analysis]: https://proceedings.neurips.cc/paper/2020/file/1f47cef5e38c952f94c5d61726027439-Paper.pdf

论文提到的方法：”经典的方法是**扩展标准的单变量模型**，**产生向量自回归**、**多变量GARCH**和**多变量状态空间模型**“



对应ADF检验，多变量中的平稳性检测：Johansen

预测方法：VAR（Vector Auto Regression 向量自回归）

方法概述：https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/



## 孙文麒

### Informer2020

- Repository: [Informer2020](https://github.com/zhouhaoyi/Informer2020)
- 需求前置
  - Python 3.6
  - matplotlib == 3.1.1
  - numpy == 1.19.4
  - pandas == 0.25.1
  - scikit_learn == 0.21.3
  - torch == 1.4.0
- 优势
  - 论文实现 (AAAI'21 Best Paper)
  - 提供样例数据集
  - 支持使用colab运行样例
  - 有能力对长时间进行预测
  - 详细的文档
  - 持续运维 (last update in Feb 22, 2021)
- 劣势
  - 模型较为复杂

### latent_ode

- Repository: [latent_ode](https://github.com/YuliaRubanova/latent_ode)
- 需求
  - torchdiffeq
- 优势
  - 在公用数据集上运行
  - 自动下载数据集
- 缺陷
  - 主要针对 irregularly-sampled(非定期取样) 时序预测
  - 没有提供用于预测的接口

## 姚凯楠

### Paper

[CoFlux:]: https://netman.aiops.org/wp-content/uploads/2019/05/CoFlux_camera-ready1.pdf

1. 摘要

   对于多维时间序列主要存在以下两个问题：

   1. 对于不同结构的特征，如何将他们从正常的特征中区分出来

      这里涉及到的问题的单指标的情况下一样，要考虑seasonal trend stationary

   2. 对于两个KPI来说，判断他们之间的关系主要要考虑三点：

      1. 是否关联波动
      2. 波动的时间顺序
      3. 波动的方向性

      一般来说先确定1然后再去确定23。

   但是在这篇文章中主要的部分都是在论述相关性的问题，对于区分涉及的情况比较少。论文的结论是CoFlux效果比别的算法好，在实际应用上可以对于警告压缩、找出最相关的几个kpi、构建异常传播链。

2. 在这个算法中，我们定义相关性是使用fluctuations 而不是kpi，kpi的直接值不能反应真实的相关性。

   Flux-features的定义是 通过对于kpi采样得到的值以及预测值的误差所得到的一个序列，然后对于两个kpi的序列检测是否相关，以及时序性和方向性

   所以对于预测模型的选取就非常重要，文章中采用了多种模型以及使用经验调参得到预测值进而计算Flux-features，其中还包括了特征的放大。


​		然后将放大后的特征值作为输出输入到Correlation measurement算法中，对于一对flux-features的放大后的值两两之间计算FCC函数，然后通过ccv以及shiftV定义他们的时序性和方向性。具体Correlation measurement和FCC的算法再文论中有，篇幅很长就不放上来了。

3. 存在的几个问题的说明

   1. 为什么要采用86个detector，对于使用的两个数据集，这两个数据集的特性不是很相同，在这个过程中，并没有哪个detector占据了绝对性的主导地位，最高也就20% ，大多数在10%以下，所以无法只使用部分detector。但是在这里存在一个猜想，就是是否存在部分占据百分比很少的detector然后我们可以选择不用它们提升效率，但是这需要大量的数据证明。

   2. 特征放大可以明显的强化算法的表现，所以放大这一步也是不可或缺的。

   3. 关于计算效率，基本上时间和kpi长度成正比，如果detector的数量固定，可以用快速傅里叶变换加速

      

### DEMO

Paper：Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks

[paper]: https://arxiv.org/pdf/1703.07015.pdf
[code]: https://github.com/laiguokun/LSTNet
[DataSet]: https://github.com/laiguokun/multivariate-time-series-data	" 没有metadata我无法解释"

这篇文章主要是针对长短周期（一天和一周，类似于上文中diff中的lastday和lastweek）做的预测，但是没有考虑到季节性等因素。

https://dl.acm.org/doi/10.1145/3394486.3403118

https://github.com/nnzhan/MTGNN



### EXTRA

别人整理的会议（ 多维时间序列间的异常关联与因果推断）：https://blog.csdn.net/weixin_53741275/article/details/111973738

https://mp.weixin.qq.com/s?__biz=Mzg3NDUwNTM3MA==&mid=2247485139&idx=1&sn=8310a0a9dcd10bd0e0eb6d87deba8ece&chksm=cecef326f9b97a30205d102114c4e37da827931a1c27c1d3cfcba5bd29e9c1dd86fb78998ada&scene=178&cur_album_id=1573418835687309313#rd



