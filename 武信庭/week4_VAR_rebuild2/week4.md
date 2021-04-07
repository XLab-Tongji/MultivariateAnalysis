# 武信庭-第四周-VAR_rebuild2

## 本周进度

继续修改VAR模型

### 1. VAR方法修改

修改为循环增量训练，每次预测一步

```python
train_len = int(0.8 * (len(data)))
left_len = int(0.2 * (len(data)))
# 创建训练集与验证集
train = data[:train_len]
valid = data[train_len:]
print("init train df is: ", train)
print("valid length is: ", left_len)

for i in range(left_len):
    # 模型训练
    model = VAR(endog=train)
    model_fit = model.fit()
    cols = data.columns

    # 在验证集上进行一步预测
    prediction = model_fit.forecast(model_fit.y, steps=1)
    # print("prediction is: ", prediction)

    ind = valid.index[i]
    # 将预测结果转为df格式并加入训练集
    train.loc[ind] = prediction[0]
    # print(train)

print("final train is:", train)
pred = train.copy(deep=True)

for i in range(0, 8):
    print('rmse value for', cols[i], 'is : ', sqrt(mean_squared_error(pred.iloc[i], valid.iloc[i])))
```

结果：

```
final train is:                            dew       temp  ...      rain   pollution
date                                       ...                      
2010-01-02 00:00:00 -16.000000  -4.000000  ...  0.000000  129.000000
2010-01-02 01:00:00 -15.000000  -4.000000  ...  0.000000  148.000000
2010-01-02 02:00:00 -11.000000  -5.000000  ...  0.000000  159.000000
2010-01-02 03:00:00  -7.000000  -5.000000  ...  0.000000  181.000000
2010-01-02 04:00:00  -7.000000  -5.000000  ...  0.000000  138.000000
...                        ...        ...  ...       ...         ...
2014-12-31 19:00:00   1.779621  12.168487  ...  0.209939   93.251906
2014-12-31 20:00:00   1.779621  12.168487  ...  0.209939   93.251906
2014-12-31 21:00:00   1.779621  12.168487  ...  0.209939   93.251906
2014-12-31 22:00:00   1.779621  12.168487  ...  0.209939   93.251906
2014-12-31 23:00:00   1.779621  12.168487  ...  0.209939   93.251906

[43800 rows x 8 columns]
rmse value for dew is :  62.54004327229075
rmse value for temp is :  61.45001261187828
rmse value for press is :  62.25672905397457
rmse value for wnd_dir is :  65.7630239002131
rmse value for wnd_spd is :  21.802320977363856
rmse value for snow is :  8.977293578802021
rmse value for rain is :  7.433035214500197
rmse value for pollution is :  18.814345922726094
```

