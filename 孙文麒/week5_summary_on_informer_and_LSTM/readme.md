# Week5 Summary on Informer and LSTM

## 1 Informer

### 1.1 简介

#### 什么是Informer?

- Informer，全称Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting 也就是针对长时序预测的高效率Transformer
- Transformer是2017年的一篇论文《Attention is All You Need》提出的一种模型架构，现在被广泛应用于NLP的各个领域
- 特点
  - 大幅降低注意力机制的时间与空间复杂度
  - 可以处理超长输入序列
  - 提升长序列预测的速度

#### Informer是如何做到这些特点的

- 提出一种用于时间序列预测的高效自注意力机制
- 由传统的一次预测一个值转为直接预测序列
- 优化的Encoder-Decoder结构

### 1.2 训练及预测过程

- Informer的几个输入参数
  - seq_len 输入encoder的长度
  - label_len 输入decoder的已知部分长度
  - pred_len 需要预测的长度
  - 最终输入decoder的序列为`concat[start token series(label_len), zero padding series(pred_len)]`


- 我们假设现在需要预测t时刻之后，维度为7的向量，设定参数为 `seq_len=72, label_len=36 pred_len=24` 时间间隔为1小时
  - batch_x
    - t时刻前3天的数据，是一个7维的list，长度为`seq_len + pred_len`
  - batch_y
    - t时刻后1天的数据，是一个7维的list，长度为`label_len + pred_len`
  - 首先将batch_x全部输入encoder，得到feature map （与transformer类似）
  - 然后拼接 `concat[batch_x后36个, 000...00(24位) ]` 因此其长度为 36+24=60 将其作为decoder的输入
  - 最后输出时，将后24位的0替换成预测输出，作为输出结果
  - 在预测过程中不采用传统的滑动窗口，而是利用Multi-head Attention+全连接层一次性预测出结果。

### 1.3 注意事项

- 更换数据集时
  - 第一列必须是timestamp格式
  - 按照时间选择间隔，天、时、分
  - target维必须位于最后一列，并修改target与数据列一致
  - 修改enc_in和dec_in，与数据维度一致
- 调参
  - 主要需要调整seq_len, label_len, pred_len几个参数, 符合数据特征。

### 1.4 访问方法

- 由于173服务器近期显卡负载较大，推荐使用Google Colab打开ipynb
  - 从Informer github主页 进入 Colab Example
  - 设置Colab 使用GPU
  - 执行代码
- 也在173服务器上搭建好了，可以进入训练部分，但是最近显存不足

### 1.5 实验结果

> 本部分实验结果都是基于StandardScaler，在pollution上的预测结果，MSE为对多个维度的综合损失计算

| 优化方案                           | MSE    |
| ---------------------------------- | ------ |
| 无优化，不进行调参                 | 0.5627 |
| 优化维度，删除wnd_dir和wind snow列 | 0.2799 |
| 优化维度，输入参数下调             | 0.2629 |

## 2 LSTM

### 2.1 简介

- 在多变量预测任务中，通常采用CNN+LSTM或RNN+LSTM的模型
- CNN/RNN 进行特征提取，LSTM进行预测
- 部分还在此基础上加入了注意力机制，用于处理多维数据
- 通常只能预测单步，利用滑动窗口预测更多

### 2.2 训练及预测过程

- 常见的几个输入参数
  - INPUT_DIMS 输入维度
  - TIME_STEPS 输入序列长度
  - 部分模型支持长序列预测 会有 PRED_LEN
- 对于单步预测
  - 采取前TIME_STEPS预测后一步的方法
- 对于多步预测
  - 基于单步预测，使用teacher force，即将预测数据作为预测下一步的已知数据

### 2.3 注意事项

- 更换数据集时
  - 注意维度变化
  - 修改load_csv的部分
- 调参
  - 调整上述几个输入参数即可

### 2.4 访问方法

- 在173服务器上搭建好了 位于tunnelKG container之中
执行以下命令
```sh
docker exec -it tunnelKG bash
cd time_series/Forecasting-on-Air-pollution-with-RNN-LSTM/
jupyter notebook --port 12001 --ip 0.0.0.0 --allow-root
```
通过浏览器访问http://10.60.38.173:12001/即可

### 2.5 实验结果

> 本部分实验结果都是在pollution上的单步预测结果，MSE为对目标维度的综合损失计算

| 优化方案                           | MSE/RMSE    |
| ---------------------------------- | ------ |
| 基于MinMaxScaler(1,0)                 | MSE:8.9546e-04 |
| 基于MinMaxScaler(1,0)，进行inverse变换 | RMSE:18 |
| 基于StandardScaler             | MSE:0.55 |