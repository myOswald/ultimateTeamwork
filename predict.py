# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# %%[markdown]
# ## 我们先看看数据有哪些列，然后将日期作为索引并且去掉索引列
import akshare as ak
stock_zh_a_daily_hfq_df = ak.stock_zh_a_daily(symbol="sh600582", adjust="hfq")
stock_zh_a_daily_hfq_df.to_csv('gupiao.csv')
print(stock_zh_a_daily_hfq_df)
# %%
df = pd.read_csv('gupiao.csv')
columns = df.columns
print(columns)
df = df.set_index(df['date']).drop(columns=['date'])

# %%[markdown]
# ## 定义函数，得到rsi参数，以便后续使用

# %%
def cal_rsi(df0, period=6):  # 默认周期为6日

    """
    df0为整理后的表格，
    period为周期，默认为6。
    """

    df0['diff'] = df0['close'] - df0['close'].shift(1)  # 用diff存储两天收盘价的差
    df0['diff'].fillna(0, inplace=True)  # 空值填充为0
    df0['up'] = df0['diff']  # diff赋值给up
    df0['down'] = df0['diff']  # diff赋值给down
    df0['up'][df0['up'] < 0] = 0  # 把up中小于0的置零
    df0['down'][df0['down'] > 0] = 0  # 把down中大于0的置零
    df0['avg_up'] = df0['up'].rolling(period).sum() / period  # 计算period天内平均上涨点数
    df0['avg_down'] = abs(df0['down'].rolling(period).sum() / period)  # 计算period天内平均下降点数
    df0['avg_up'].fillna(0, inplace=True)  # 空值填充为0
    df0['avg_down'].fillna(0, inplace=True)  # 空值填充为0
    df0['rsi'] = 100 - 100 / (1 + (df0['avg_up'] / df0['avg_down']))  # 计算RSI
    return df0  # 返回原DataFrame

# %%
df = cal_rsi(df)
df = df.dropna()

# %%[markdown]
# ## 定义函数，得到经济学参数KDJ

# %%
def cal_KDJ(df):
    import pandas as pd

    df['lowest'] = df['low'].rolling(9).min()
    df['lowest'].fillna(value=df['low'].expanding().min(), inplace=True)
    df['highest'] = df['high'].rolling(9).max()
    df['highest'].fillna(value=df['high'].expanding().max(), inplace=True)
    df['RSV'] = (df['close'] - df['lowest']) / (df['highest'] - df['lowest']) * 100

    df['K'] = pd.DataFrame(df['RSV']).ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    return df

# %%
df = cal_KDJ(df)
df = df.dropna()

# %%[markdown]
# ## 得到样本训练集，测试集的切分

# %%
def get_n_train_val(df, ts=0.7, vs=0.9):
    """
   获取df的长度n，
   训练集的比例，以ts代表，默认为0.7
   验证集的比例，以vs代表且vs>ts，默认为0.9
   测试集的比例。
   """

    n = len(df)
    train_split = int(ts * n)
    val_split = int(vs * n)

    return n, train_split, val_split

n, train_split, val_split = get_n_train_val(df)
print(n, train_split, val_split)
print(df)
df_origin = df

# %%[markdown]
# ## 将数据归一化操作

# %%
def standardization(df, train_split):
    "train_split为get_train_val函数中获得的训练集比例。"

    df_mean = df[:train_split].mean(axis=0)
    df_std = df[:train_split].std(axis=0)
    df = (df - df_mean) / df_std

    df = df.values
    target = df[:, 3]  # 这里的指close,到时根据所得数据取第二个数字
    return df, target, df_mean, df_std

df, target, df_mean, df_std= standardization(df, train_split)
print(type(df))
print(df.shape)

# %%[markdown]
# ## 得到训练集样本，验证集样本，测试集样本

# %%
def window_generator(dataset, target, start_index,
                     end_index, history_size, target_size):
    """
    dataset为df，target为先前选择的需要预测的值,
    start_index为开始时间，end_index为结束时间,
    history_size为过去几天长度，target_size为需要预测的天数的长度。
    """
    import pandas as pd
    import numpy as np

    features = []
    labels = []

    if end_index is None:
        end_index = len(dataset) - target_size

    start_index += history_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        features.append(dataset[indices])
        labels.append(target[i:i + target_size])

    return np.array(features), np.array(labels)


def data_processing(df, n, train_split, val_split, target, history_size=5):
    """
    df标准化后的表格，
    n为get_train_val函数中获得的df的长度，
    train_split为get_train_val函数中获得的训练集比例，
    val_split为get_train_val函数中获得的验证集比例，
    target为standardization函数中获得的预测目标，
    history_size决定以过去几天的数据为基础来预测，默认为5天。
    """
    import tensorflow as tf
    import pandas as pd

    # 以一天数据预测未来数据的特征和标签定义
    X_train_single, y_train_single = window_generator(dataset=df, target=target, start_index=0,
                                                      end_index=train_split, history_size=1, target_size=1)

    X_val_single, y_val_single = window_generator(dataset=df, target=target, start_index=train_split,
                                                  end_index=val_split, history_size=1, target_size=1)

    X_test_single, y_test_single = window_generator(dataset=df, target=target, start_index=val_split,
                                                    end_index=n - 1, history_size=1, target_size=1)

    # 以histroy_size天数据预测未来数据的特征和标签定义
    X_train_multi, y_train_multi = window_generator(dataset=df, target=target, start_index=0,
                                                    end_index=train_split, history_size=history_size, target_size=1)

    X_val_multi, y_val_multi = window_generator(dataset=df, target=target, start_index=train_split,
                                                end_index=val_split, history_size=history_size, target_size=1)

    X_test_multi, y_test_multi = window_generator(dataset=df, target=target, start_index=val_split,
                                                  end_index=n - history_size, history_size=history_size, target_size=1)

    ##########
    BUFFER_SIZE = 2000
    BATCH_SIZE = 100
    ##########

    # 将特征与标签配对
    train_single = tf.data.Dataset.from_tensor_slices((X_train_single, y_train_single))
    val_single = tf.data.Dataset.from_tensor_slices((X_val_single, y_val_single))

    train_multi = tf.data.Dataset.from_tensor_slices((X_train_multi, y_train_multi))
    val_multi = tf.data.Dataset.from_tensor_slices((X_val_multi, y_val_multi))

    # 数据增强
    train_single = train_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val_single = val_single.cache().batch(BATCH_SIZE).repeat()

    train_multi = train_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    val_multi = val_multi.cache().batch(BATCH_SIZE).repeat()

    return train_single, val_single, train_multi, val_multi, X_test_single, y_test_single, X_test_multi, y_test_multi


train_single, val_single, train_multi, val_multi, X_test_single, y_test_single, X_test_multi, y_test_multi = data_processing(
    df, n, train_split, val_split, target, history_size=5)
print('分割线---------------------------------------------------------')
print(train_single, val_single, train_multi, val_multi)

# %%[markdown]
# ## 定义神经网络参数，构建神经网络类
class lstm_Model(tf.keras.Model):
    # nums_of_dense表示每层的神经元
    def __init__(self, nums_of_lstm=32, nums_of_dense1=3, nums_of_dense2=1):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(nums_of_lstm, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(nums_of_dense1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(nums_of_dense2)

    def call(self, inputs):
        z = self.lstm(inputs)
        z = self.dense1(z)
        z = self.dense2(z)
        return z

lstm = lstm_Model(32, 4, 1)

# %%[markdown]
# ## 构建神经网络运行所需要的损失曲线函数和拟合预测函数
def loss_curve(history):
    "绘制损失曲线,history为compile_and_fit训练之后的模型"

    import matplotlib.pyplot as plt

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def compile_and_fit(model, train_data, val_data, patience=10):
    """
    model是之前选择的模型，
    train_data是之前训练用的输入，如train_single,
    val_data是之前测试用的输入，如val_single。
    """
    #############
    EPOCHS = 100
    EVALUATION_INTERNAL = 120
    #############

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=1,
        patience=patience,
        mode='auto',
        restore_best_weights=True)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0),
                  loss='mae')

    history = model.fit(train_data, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERNAL,
                        validation_steps=50,
                        validation_data=val_data,
                        callbacks=[early_stopping])
    return history

lstm_history = compile_and_fit(lstm, train_multi, val_multi)

loss_curve(lstm_history)

# %%[markdown]
# ## 完成神经网络的预测并绘制图形

# %%
lstm_results = lstm.predict(X_test_multi)

fig = plt.figure(figsize=(15, 8))
ax = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=3)
ax.xaxis.set_major_locator(mticker.MaxNLocator(10))

plt.plot(y_test_multi, label='oringin')
plt.plot(lstm_results, label='lstm')
plt.legend()

plt.show()

# %%[markdown]
# ## 将得到的神经网络预测，之前的rsi，KDJ等参数与原有的表绘制于一张新表中

# %%
lstm = np.full((len(df), 1), np.nan)
for i in range(len(df)-len(lstm_results), len(df)):
    lstm[i] = lstm_results[i - len(df) + len(lstm_results)] * df_std[3] +df_mean[3]
print(len(lstm))
print(len(df_origin))
data = pd.DataFrame(lstm, columns=['predict'])
data.index = df_origin.index
data = pd.concat([df_origin, data], axis=1)
data.to_csv('new.csv')
# %%
