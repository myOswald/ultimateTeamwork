# 2020-8-大作业

## 1、团队总体策略

团队经过开会讨论后，决定不限于选取一只股票，最终选取天地科技、东风汽车、中国银行等公司的股票进行神经网络的预测和分析，制定了五种策略，确保收益最大化，团队成员分工明确，大家各司其职。

## 2、组员分工

陈 俊：比较多种数据并选取合适的数据和负责写readme

杨胜龙：对选取的数据进行处理和做答辩PPT

帅逸轩：搭建神经网络模型，调整神经网络的参数并最优化神经网络

张从佳：进行策略分析，比较多只股票和多种策略，确保收益最大化，带着我们一起赚钱

## 3、代码基本介绍

### 选取数据

获取的天地sh天地科技的股票数据，并导出为csv文件，后面在做回测的时候，其他股票的数据也用同样的方法获取。

```
import akshare as ak
stock_zh_a_daily_hfq_df = ak.stock_zh_a_daily(symbol="sh600582", adjust="hfq")
stock_zh_a_daily_hfq_df.to_csv('gupiao.csv')
print(stock_zh_a_daily_hfq_df)
```

### 数据处理

这里将数据处理过程封装成了多个函数（函数代码见代码文件)，分别设置验证集、测试集比例、归一化处理和预测目标等，对获取的数据添加RSI参数、KDJ参数多个特征，为后续神经网络的预测提高更高的准确性。

```
def cal_rsi(df0, period=6): 
def cal_KDJ(df):
def get_n_train_val(df, ts=0.7, vs=0.9):
def standardization(df, train_split):
def window_generator(dataset, target, start_index,end_index, history_size, target_size):
def data_processing(df, n, train_split, val_split, target, history_size=5):
```

### 搭建神经网络以及设置神经网络

搭建神经网络，设置全连接层的个数，绘制损失曲线以及训练模型

```
class lstm_Model(tf.keras.Model):
def loss_curve(history):
def compile_and_fit(model, train_data, val_data, patience=10):
```

### 神经网络预测和分析

将搭建好的神经网络进行预测并绘制图形，将得到的神经网络预测，之前的RSI，KDJ等参数与原有的表绘制于一张新表中

### 数据回测

对得到的RSI、KDJ等股票的各种因子值以及下一天价格的预测值进行回测，回测的初始资本为100万元，回测时间为最近的一年，根据不同的策略，根据比较得到最优的策略。

```
class SingleBackTest(object):
```

## 4、最终策略及收益

制定了以下五种策略：

1. 根据rsi指数制定方案。rsi的全称是相对强弱指标，通常以30和70作为判断依据。当rsi小于30时，说明股票的价格即将上涨， 此时对于做空而言就要进行平仓；当rsi大于70时，说明股票 快要涨停了，即将开始下跌，此时对于做多而言就要进行平仓。

2. 根据KDJ指数中的KD制定方案。上涨趋势中，K值小于D值，K线向上突破D线时，为买进信号。下跌趋势中，K值大于D值，K线向下跌破D线时，为卖出信号。

3. 根据KDJ指数制定方案。当K，D，J都小于20时，适合买入；当K，D，J都大于80时，适合卖出。

4. 根据预测的股价pred制定方案。当预测到未来一天的股价即将上涨时，适合买入；当预测到未来一天的股价即将下跌时，适合卖出。

5. 根据平均值曲线avg制定方案。当短期平均（这里取5天）超过长期平均（这里取25天）时，适合买入；当长期平均 超过短期平均时，适合卖出。

每一种方案又分为做多和做空两种，做多就是先买后卖，做空就是先卖后买，这样一来就产生了10种策略， 依次对10种策略进行回测，目的是找出收益最多的一种方案。

根据结果显示，天地科技的根据KDJ指数设定的方案可以实现59万的收益，实现收益最大化。

## 5、运行本代码可能需要安装的第三方库

```
tensorflow
akshare
```