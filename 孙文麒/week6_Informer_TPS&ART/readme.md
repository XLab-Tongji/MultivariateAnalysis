# Week6 Informer on TPS&ART dataset

## 1 Test1: 30min prediction with 60min input length within 3 months on multivariate

### 1.1 Data preprocess

采取每分钟1个数据点，取3个月的数据

|Rows|Value|
|----------|-----|
|total rows|129600|
|train|90631|
|val|12931|
|test|25891|

### 1.2 Hyper Parameters
|Parameters|Value|
|----------|-----|
|dimension|2|
|seq_len|60|
|label_len|30|
|pred_len|30|
|inverse|False|
|others|default|

### 1.3 Result

未做inverse变换
|Evaluation|Value|
|----------|---------|
|speed|0.1429s/iter|
|time cost per Epoch|406s|
|MSE|0.07224|
|MAE|0.04941|

![TPS_1_2100](./images/TPS_1_2100.png)
![TPS_1_3000](./images/TPS_1_3000.png)

## 2 Test2: 1d prediction with 2d input length within 3 months on univariate

### 2.1 Data preprocess

采取每10分钟1个数据点，取3个月的数据

|Rows|Value|
|----------|-----|
|total rows|12960|
|train|8641|
|val|1153|
|test|2449|

### 2.2 Hyper Parameters
|Parameters|Value|
|----------|-----|
|dimension|2|
|seq_len|288|
|label_len|144|
|pred_len|144|
|inverse|False|
|features|S|
|others|default|

### 2.3 Result

|Evaluation|Value|
|----------|---------|
|speed|0.2858s/iter|
|time cost per Epoch|75.77s|
|MSE|0.11456|
|MAE|0.22674|

![TPS_2_0](./images/TPS_2_0.png)

## 3 Test3: 2d prediction with 2d input length within 3 months on univariate

### 3.1 Data preprocess

采取每10分钟1个数据点，取3个月的数据

|Rows|Value|
|----------|-----|
|total rows|12960|
|train|8641|
|val|1153|
|test|2449|

### 3.2 Hyper Parameters
|Parameters|Value|
|----------|-----|
|dimension|2|
|seq_len|288|
|label_len|144|
|pred_len|288|
|inverse|False|
|features|S|
|others|default|

### 3.3 Result

|Evaluation|Value|
|----------|---------|
|speed|0.7545s/iter|
|time cost per Epoch|198.46s|
|MSE|0.5007|
|MAE|0.5720|

![TPS_3_0](./images/TPS_3_0.png)

## 4 Test4: 2d prediction with 3d input length within 3 months on univariate

### 4.1 Data preprocess

采取每10分钟1个数据点，取3个月的数据

|Rows|Value|
|----------|-----|
|total rows|12960|
|train|8641|
|val|1153|
|test|2449|

### 4.2 Hyper Parameters
|Parameters|Value|
|----------|-----|
|dimension|2|
|seq_len|432|
|label_len|144|
|pred_len|288|
|inverse|False|
|features|S|
|others|default|

### 4.3 Result

|Evaluation|Value|
|----------|---------|
|speed|0.9403s/iter|
|time cost per Epoch|246.80s|
|MSE|0.1707|
|MAE|0.2560|

![TPS_4_0](./images/TPS_4_0.png)

## 5 修改预测思路

- 阅读`data_loader`源码发现，Informer中并未提供按照原有格式输出预测的接口，需要手动实现。
- Informer的预测可以将一个csv文件作为输入，输出一个2-D List `preds`，第一维是timestamp对应预测值（0~pred_len），第二维代表预测项。
- 实现`predict_append`思路如下：
  - 调用`read_csv`获取待预测数据，并提取最后一个`timedate`，获取一个`time_interval`作为参数。
  - 将`time_interval`与`last_date`结合，生成`pred_len`个`timedate`作为`timedate`维的数据。
  - 拼接在preds的第二维中，再逐`timedate`将数据append到原有数据后，保存。

## 6 TODO

- 将修改后的数据\(即去除5:59:00和6:00:00的重复项\)进行处理，按照目前参数进行再次实验，更新结果
- 按照当前思路，实现`predict_append`方法，并将结果可视化。