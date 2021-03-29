# Week3 Model Optimize & Seasonality Effect

## LSTM

**上周遗留的问题**：

由于归一化方法不一致，导致LSTM与Informer的指标差异较大

**问题原因**

- LSTM使用的是`sklearn.preprocess.MinMaxScaler`，且设置范围为(0,1)
- Informer使用的是`sklearn.preprocess.StandardScaler`

**调整**：

在将LSTM转为`StandardScaler`后，其预测准确度有明显降低，进行`inverse_transform`后的RMSE达到了65，训练过程中的loss也只有0.55，明显低于Informer。

![image-20210329102358155](img\LSTM_loss.png)

![image-20210329102324962](img\LSTM_pred.png)

## Informer

### 思路1

#### 问题描述：

数据中存在维度，其值对于预测pollution作用可能较小，或甚至会起反效果。例如wnd_dir，描述了风向，难以将其转化为数字，如果按照方向映射为数字，会造成各个风向之间的关系在此过程中损失掉。

#### 方法描述：

- 删除了数据中的`wnd_dir`、`snow`、`rain`的三项参数

#### 结果：

- ![image-20210324144852411](img\Informer.png)

```sh
mse:0.2799077189809823, mae:0.31754937508108366
```

- 模型预测结果有了较大改善

#### 总结分析：

- 删除这三列后对模型效果产生了大幅度优化，换而言之，该数据集中的维度相关性并不是很强，而Informer的精度会受制于数据集的无关维度。
- 后续在其他训练集上训练时也应当留意这一问题，有必要对数据集进行预处理，筛选合适的维度。



### 思路2：

#### 问题描述：

模型参数主要是针对ETDataset的，和pollution数据集存在差异。并且在调参过程中存在过拟合问题，经常在训练1个epoch后，就会产生trainning loss下降，vali loss上升的问题。

#### 方法描述：

首先将输入长度下调

```python
args.seq_len = 48 # input sequence length of Informer encoder
args.label_len = 24 # start token length of Informer decoder
args.pred_len = 24 # prediction sequence length
```

随后再降低learning_rate

```python
args.learning_rate = 0.00005
```

#### 结果描述：

- 在训练过程中，过拟合问题仍然没有解决，模型依然在训练了1个epoch后产生early-stopping计数

- 精度有小幅提升

```sh
mse:0.2629131078720093, mae:0.3101675808429718
```

