# Week7 Library Exploration

## 1 Related libraries

### 1.1 Darts

**Introduction:**
darts is a Python library for easy manipulation and forecasting of time series. It contains a variety of models, from classics such as ARIMA to deep neural networks. The models can all be used in the same way, using `fit()` and `predict()` functions, similar to scikit-learn. The library also makes it easy to backtest models, and combine the predictions of several models and external regressors. Darts supports both univariate and multivariate time series and models, and the neural networks can be trained on multiple time series.

**Descriptions:**

- 只提供了一个双变量预测的模型，其余基本都是单变量
- pre-trained model&covariates

### 1.2 TensorFlow-Time-Series-Examples

**Introduction:**

- Time Series Prediction with tf.contrib.timeseries

**Descriptions:**

- 提供了多种预测方法
- 多变量的在文档里只有LSTM

## 2 less related libraries

|Project Name|Description|
|------------|-----------|
|Arrow|Converting dates&times|
|cesium|feature extraction|
|flint|Time series library for Apache Spark|
|prophet|univarate with multiple seasonality|

## 3 Conclusion

- 几乎所有的库都不是面向多变量预测任务的
- 部分库虽然是有关变量预测的，但是普遍版本较老
- 极少部分库具有完整的阐述多变量预测的文档
- 综上，可能没有适用于我们数据集的多变量时间预测库