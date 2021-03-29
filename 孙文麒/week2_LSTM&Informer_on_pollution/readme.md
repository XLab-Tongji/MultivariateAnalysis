## Result

### Infomer

https://github.com/zhouhaoyi/Informer2020
- 实验总结
  - MSE结果大约在0.5~0.6左右（未调优）如果去除部分列，例如wnd_dir后能降到约0.3~0.4
  - 符合论文结果
  - 但是MSE计算与其他方法似乎有所差异，大致看了一下代码，是用了自己的一套归一化方法。

#### MS

```sh
Use GPU: cuda:0
>>>>>>>start training : informer_custom_ftMS_sl96_ll48_pl24_dm512_nh8_el3_dl2_df512_atprob_ebtimeF_dtTrue_exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 30540
val 4358
test 8737
	iters: 100, epoch: 1 | loss: 0.5228176
	speed: 0.3655s/iter; left time: 2056.0753s
	iters: 200, epoch: 1 | loss: 0.5498728
	speed: 0.3705s/iter; left time: 2047.0255s
	iters: 300, epoch: 1 | loss: 0.5300557
	speed: 0.3728s/iter; left time: 2022.5514s
	iters: 400, epoch: 1 | loss: 0.4600641
	speed: 0.3725s/iter; left time: 1983.5159s
	iters: 500, epoch: 1 | loss: 0.5551958
	speed: 0.3721s/iter; left time: 1944.4590s
	iters: 600, epoch: 1 | loss: 0.5077974
	speed: 0.3723s/iter; left time: 1908.1805s
	iters: 700, epoch: 1 | loss: 0.7335935
	speed: 0.3722s/iter; left time: 1870.3047s
	iters: 800, epoch: 1 | loss: 0.5412961
	speed: 0.3723s/iter; left time: 1833.4413s
	iters: 900, epoch: 1 | loss: 0.8898655
	speed: 0.3723s/iter; left time: 1796.3553s
Epoch: 1, Steps: 954 | Train Loss: 0.6078118 Vali Loss: 0.4658842 Test Loss: 0.5615195
Validation loss decreased (inf --> 0.465884).  Saving model ...
Updating learning rate to 0.0001
	iters: 100, epoch: 2 | loss: 0.5294149
	speed: 1.1987s/iter; left time: 5598.9720s
	iters: 200, epoch: 2 | loss: 0.4471902
	speed: 0.3731s/iter; left time: 1705.5011s
	iters: 300, epoch: 2 | loss: 0.4094632
	speed: 0.3736s/iter; left time: 1670.4417s
	iters: 400, epoch: 2 | loss: 0.7282873
	speed: 0.3734s/iter; left time: 1632.2135s
	iters: 500, epoch: 2 | loss: 0.4496378
	speed: 0.3737s/iter; left time: 1595.9957s
	iters: 600, epoch: 2 | loss: 0.6764847
	speed: 0.3740s/iter; left time: 1559.9215s
	iters: 700, epoch: 2 | loss: 0.3197503
	speed: 0.3736s/iter; left time: 1520.9571s
	iters: 800, epoch: 2 | loss: 0.3826064
	speed: 0.3737s/iter; left time: 1484.0811s
	iters: 900, epoch: 2 | loss: 0.3440326
	speed: 0.3737s/iter; left time: 1446.7399s
Epoch: 2, Steps: 954 | Train Loss: 0.4918885 Vali Loss: 0.5781804 Test Loss: 0.7230647
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
	iters: 100, epoch: 3 | loss: 0.5021367
	speed: 1.2034s/iter; left time: 4473.0585s
	iters: 200, epoch: 3 | loss: 0.4213963
	speed: 0.3742s/iter; left time: 1353.3446s
	iters: 300, epoch: 3 | loss: 0.2029914
	speed: 0.3738s/iter; left time: 1314.6892s
	iters: 400, epoch: 3 | loss: 0.3528716
	speed: 0.3756s/iter; left time: 1283.3930s
	iters: 500, epoch: 3 | loss: 0.4068549
	speed: 0.3760s/iter; left time: 1247.2145s
	iters: 600, epoch: 3 | loss: 0.4418854
	speed: 0.3763s/iter; left time: 1210.4123s
	iters: 700, epoch: 3 | loss: 0.4507042
	speed: 0.3762s/iter; left time: 1172.6966s
	iters: 800, epoch: 3 | loss: 0.2281347
	speed: 0.3761s/iter; left time: 1134.8403s
	iters: 900, epoch: 3 | loss: 0.2311412
	speed: 0.3761s/iter; left time: 1097.1004s
Epoch: 3, Steps: 954 | Train Loss: 0.3549121 Vali Loss: 0.5646082 Test Loss: 0.7329246
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
	iters: 100, epoch: 4 | loss: 0.3110434
	speed: 1.2114s/iter; left time: 3347.0433s
	iters: 200, epoch: 4 | loss: 0.2042857
	speed: 0.3762s/iter; left time: 1001.7846s
	iters: 300, epoch: 4 | loss: 0.2445684
	speed: 0.3760s/iter; left time: 963.7097s
	iters: 400, epoch: 4 | loss: 0.3267238
	speed: 0.3756s/iter; left time: 925.0868s
	iters: 500, epoch: 4 | loss: 0.1898751
	speed: 0.3761s/iter; left time: 888.7501s
	iters: 600, epoch: 4 | loss: 0.2791213
	speed: 0.3760s/iter; left time: 850.7915s
	iters: 700, epoch: 4 | loss: 0.1983039
	speed: 0.3763s/iter; left time: 813.8313s
	iters: 800, epoch: 4 | loss: 0.2223299
	speed: 0.3758s/iter; left time: 775.3317s
	iters: 900, epoch: 4 | loss: 0.2395126
	speed: 0.3759s/iter; left time: 737.9185s
Epoch: 4, Steps: 954 | Train Loss: 0.2688216 Vali Loss: 0.5729340 Test Loss: 0.7272407
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>>testing : informer_custom_ftMS_sl96_ll48_pl24_dm512_nh8_el3_dl2_df512_atprob_ebtimeF_dtTrue_exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 8737
test shape: (273, 32, 24, 8) (273, 32, 24, 1)
test shape: (8736, 24, 8) (8736, 24, 1)
mse:0.5627391981093997, mae:0.5105207450819185
```



#### M

```sh
Use GPU: cuda:0
>>>>>>>start training : informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el3_dl2_df512_atprob_ebtimeF_dtTrue_exp_0>>>>>>>>>>>>>>>>>>>>>>>>>>
train 30540
val 4358
test 8737
	iters: 100, epoch: 1 | loss: 0.4026234
	speed: 1.0306s/iter; left time: 5797.2750s
	iters: 200, epoch: 1 | loss: 0.3560256
	speed: 1.0578s/iter; left time: 5844.6173s
	iters: 300, epoch: 1 | loss: 0.6651701
	speed: 1.0702s/iter; left time: 5805.8141s
	iters: 400, epoch: 1 | loss: 0.3118287
	speed: 1.0765s/iter; left time: 5732.2732s
	iters: 500, epoch: 1 | loss: 0.6864231
	speed: 1.0783s/iter; left time: 5634.3632s
	iters: 600, epoch: 1 | loss: 0.4231052
	speed: 1.0787s/iter; left time: 5528.4250s
	iters: 700, epoch: 1 | loss: 0.2923041
	speed: 1.0791s/iter; left time: 5422.3523s
	iters: 800, epoch: 1 | loss: 1.0167510
	speed: 1.0793s/iter; left time: 5315.6623s
	iters: 900, epoch: 1 | loss: 0.3166045
	speed: 1.0810s/iter; left time: 5215.8814s
Epoch: 1, Steps: 954 | Train Loss: 0.4959290 Vali Loss: 0.3864383 Test Loss: 0.4007513
Validation loss decreased (inf --> 0.386438).  Saving model ...
Updating learning rate to 0.0001
	iters: 100, epoch: 2 | loss: 0.2580972
	speed: 3.2053s/iter; left time: 14972.0696s
	iters: 200, epoch: 2 | loss: 0.5714380
	speed: 1.0787s/iter; left time: 4930.8289s
	iters: 300, epoch: 2 | loss: 0.3493462
	speed: 1.0785s/iter; left time: 4822.0890s
	iters: 400, epoch: 2 | loss: 0.3516925
	speed: 1.0781s/iter; left time: 4712.5723s
	iters: 500, epoch: 2 | loss: 0.6990944
	speed: 1.0787s/iter; left time: 4607.1462s
	iters: 600, epoch: 2 | loss: 0.3491930
	speed: 1.0799s/iter; left time: 4504.4624s
	iters: 700, epoch: 2 | loss: 0.7676597
	speed: 1.0790s/iter; left time: 4392.6492s
	iters: 800, epoch: 2 | loss: 0.4557243
	speed: 1.0794s/iter; left time: 4286.2031s
	iters: 900, epoch: 2 | loss: 0.3278763
	speed: 1.0796s/iter; left time: 4179.2518s
Epoch: 2, Steps: 954 | Train Loss: 0.3984649 Vali Loss: 0.3940365 Test Loss: 0.4132988
EarlyStopping counter: 1 out of 3
Updating learning rate to 5e-05
	iters: 100, epoch: 3 | loss: 0.3336034
	speed: 3.2029s/iter; left time: 11905.1844s
	iters: 200, epoch: 3 | loss: 0.2328635
	speed: 1.0811s/iter; left time: 3910.2168s
	iters: 300, epoch: 3 | loss: 0.1987317
	speed: 1.0788s/iter; left time: 3794.1979s
	iters: 400, epoch: 3 | loss: 0.2378791
	speed: 1.0808s/iter; left time: 3693.0556s
	iters: 500, epoch: 3 | loss: 0.4808632
	speed: 1.0805s/iter; left time: 3584.0218s
	iters: 600, epoch: 3 | loss: 0.4983197
	speed: 1.0796s/iter; left time: 3473.1523s
	iters: 700, epoch: 3 | loss: 0.2972788
	speed: 1.0807s/iter; left time: 3368.5822s
	iters: 800, epoch: 3 | loss: 0.4253285
	speed: 1.0830s/iter; left time: 3267.5502s
	iters: 900, epoch: 3 | loss: 0.2696152
	speed: 1.0826s/iter; left time: 3157.9363s
Epoch: 3, Steps: 954 | Train Loss: 0.2855164 Vali Loss: 0.4046028 Test Loss: 0.4469986
EarlyStopping counter: 2 out of 3
Updating learning rate to 2.5e-05
	iters: 100, epoch: 4 | loss: 0.2343367
	speed: 3.2152s/iter; left time: 8883.6147s
	iters: 200, epoch: 4 | loss: 0.2245781
	speed: 1.0830s/iter; left time: 2884.0526s
	iters: 300, epoch: 4 | loss: 0.3452599
	speed: 1.0827s/iter; left time: 2775.0834s
	iters: 400, epoch: 4 | loss: 0.2715302
	speed: 1.0832s/iter; left time: 2667.9807s
	iters: 500, epoch: 4 | loss: 0.1913877
	speed: 1.0838s/iter; left time: 2560.9213s
	iters: 600, epoch: 4 | loss: 0.2780664
	speed: 1.0836s/iter; left time: 2452.2414s
	iters: 700, epoch: 4 | loss: 0.2148449
	speed: 1.0829s/iter; left time: 2342.3196s
	iters: 800, epoch: 4 | loss: 0.1802194
	speed: 1.0838s/iter; left time: 2235.9182s
	iters: 900, epoch: 4 | loss: 0.1818470
	speed: 1.0833s/iter; left time: 2126.5956s
Epoch: 4, Steps: 954 | Train Loss: 0.2286685 Vali Loss: 0.4144697 Test Loss: 0.4507922
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>>testing : informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el3_dl2_df512_atprob_ebtimeF_dtTrue_exp_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 8737
test shape: (273, 32, 24, 8) (273, 32, 24, 8)
test shape: (8736, 24, 8) (8736, 24, 8)
mse:0.40100106976140504, mae:0.3571412542537687
```

### LSTM

https://github.com/PatientEz/CNN-BiLSTM-Attention-Time-Series-Prediction_Keras
https://github.com/sagarmk/Forecasting-on-Air-pollution-with-RNN-LSTM

- 实验结果
  - 训练阶段MSE大约在0.001以内
  - 如果进行逆归一化操作的话，RMSE应该在18左右（LSTM+RNN）
  - 该仓库并未在test集上计算MSE

```sh
Epoch 1/10
39401/39401 [==============================] - 20s - loss: 0.0151 - val_loss: 0.0014
Epoch 2/10
39401/39401 [==============================] - 18s - loss: 0.0016 - val_loss: 0.0010
Epoch 3/10
39401/39401 [==============================] - 18s - loss: 0.0013 - val_loss: 7.3428e-04
Epoch 4/10
39401/39401 [==============================] - 18s - loss: 0.0011 - val_loss: 0.0011
Epoch 5/10
39401/39401 [==============================] - 18s - loss: 0.0011 - val_loss: 7.8320e-04
Epoch 6/10
39401/39401 [==============================] - 18s - loss: 9.9158e-04 - val_loss: 6.9831e-04
Epoch 7/10
39401/39401 [==============================] - 18s - loss: 9.4715e-04 - val_loss: 7.0603e-04
Epoch 8/10
39401/39401 [==============================] - 18s - loss: 9.1917e-04 - val_loss: 8.2625e-04
Epoch 9/10
39401/39401 [==============================] - 18s - loss: 8.9546e-04 - val_loss: 0.0012
Epoch 10/10
39401/39401 [==============================] - 18s - loss: 8.8273e-04 - val_loss: 0.0012
```

