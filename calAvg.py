# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('ZGYH.csv')
df['5d'] = pd.Series.rolling(df['C'],window=5).mean()
df['25d'] = pd.Series.rolling(df['C'],window=25).mean()
df[['C','5d','25d']].plot(grid=True)

df['s-l'] = df['5d'] - df['25d']
df['signal'] = np.where(df['s-l']>0,1,0)
df['signal'] = np.where(df['s-l']<0,-1,df['signal'])

df['market'] = df['C'].pct_change(1)
df['strategy'] = df['signal'].shift(1)*df['market']
df.to_csv('ZGYH.csv')

rtn=[]
for i in range(0,df.shape[0]):
    if df['signal'][i] == 1:
        rtn.append(df['market'][i])

result = np.cumsum(rtn)
plt.plot(result)
