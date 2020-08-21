# %% [markdown]
# # 回测分析
# 经过前面的处理，我们已经得到了股票或期货的各种因子值以及下一天价格的预测值，下面的代码就是利用这些计算结果进行回测分析。
# 
# 回测的初始资本为100万元，回测时间为最近的一年，为了便于分析，以下所有的策略均不会出现连续买进或者连续卖出的情况，同时假设购买股票或期货的数量没有限制。
# 
# 这里一共给出了5种回测的方案，分别是
# 
# 1. 根据rsi指数制定方案。rsi的全称是相对强弱指标，通常以30和70作为判断依据。当rsi小于30时，说明股票的价格即将上涨，
# 此时对于做空而言就要进行平仓；当rsi大于70时，说明股票
# 快要涨停了，即将开始下跌，此时对于做多而言就要进行平仓。
# 
# 2. 根据KDJ指数中的KD制定方案。上涨趋势中，K值小于D值，K线向上突破D线时，为买进信号。下跌趋势中，K值大于D值，K线向下跌破D线时，为卖出信号。
# 
# 3. 根据KDJ指数制定方案。当K，D，J都小于20时，适合买入；当K，D，J都大于80时，适合卖出。
# 
# 4. 根据预测的股价pred制定方案。当预测到未来一天的股价即将上涨时，适合买入；当预测到未来一天的股价即将下跌时，适合卖出。
# 
# 5. 根据平均值曲线avg制定方案。当短期平均（这里取5天）超过长期平均（这里取25天）时，适合买入；当长期平均
# 超过短期平均时，适合卖出。
# 
# 每一种方案又分为做多和做空两种，做多就是先买后卖，做空就是先卖后买，这样一来就产生了10种策略，
# 下面的代码会依次对10种策略进行回测，目的是找出收益最多的一种方案。

import pandas as pd
import math
from bokeh.plotting import figure, show
from bokeh.layouts import column

class SingleBackTest(object):
    def __init__(self, param):
        self.test_data = param.get('data')
        self.side = param.get('side')  # 做多还是做空
        self.factor = param.get('factor')  # 回测的策略
        self.hold_days = param.get('hold_days')  # 最大持仓天数
        self.num = 0  # 单次购买股票支数
        self.result = {}
        self.acc_ret = 0
        self.day_record = {}
        self.per_record = []
        self.init_asset = 1000000  # 初始资金
        self.slippage = 0.4
        self.pos_record = {'pos': 0, 'price': 0, 'side': ''}
        self.is_trade = False
        self.last_ret = 0
        self.time_record = []
        self.has_action = False
        self.multiple_factor = 1
        self.last_k = None
        self.last_d = None
        self.last_s = None
        self.last_y = None

    def run(self):
        """
        用于运行回测的全过程，并返回回测结果
        """
        for i, kline in enumerate(self.test_data):
            if self.factor == 'rsi':
                action = self.predict_action_rsi(kline)
            elif self.factor == 'KD':
                action = self.predict_action_KD(kline)
            elif self.factor == 'KDJ':
                action = self.predict_action_KDJ(kline)
            elif self.factor == 'pred':
                action = self.predict_action_pred(kline)
            elif self.factor == 'avg':
                action = self.predict_action_avg(kline)
            self.cal_position(kline, action)
            if i == len(self.test_data) - 1:
                self.stat(kline, True)
            else:
                self.stat(kline)
            self.is_trade = False
        return self.output()

    def predict_action_rsi(self, kline):
        pred_y = kline.get(self.factor)
        if self.side == 'long':
            if not self.last_y:
                self.last_y = pred_y
                return
            if self.last_y < 30 and pred_y > 30 and self.pos_record['pos'] == 0:
                self.last_y = pred_y
                return 'buy'
            elif self.pos_record['pos'] == 1 and pred_y > 70:
                self.last_y = pred_y
                return 'close'
            self.last_y = pred_y
        else:
            if not self.last_y:
                self.last_y = pred_y
                return
            if self.last_y < 30 and pred_y > 30 and self.pos_record['pos'] == 0:
                self.last_y = pred_y
                return 'sell'
            elif self.pos_record['pos'] == 0 and pred_y > 70:
                self.last_y = pred_y
                return 'close'
            self.last_y = pred_y
        return None

    def predict_action_KD(self, kline):
        pred_k = kline.get('K')
        pred_d = kline.get('D')
        if self.side == 'long':
            if not self.last_k:
                self.last_k = pred_k
                return
            if not self.last_d:
                self.last_d = pred_d
                return
            if self.last_k < self.last_d and pred_k > pred_d and self.pos_record['pos'] == 0:
                self.last_k = pred_k
                self.last_d = pred_d
                return 'buy'
            elif self.pos_record['pos'] == 1 and self.last_k > self.last_d and pred_k < pred_d:
                self.last_k = pred_k
                self.last_d = pred_d
                return 'close'
            self.last_k = pred_k
            self.last_d = pred_d
        if self.side == 'short':
            if not self.last_k:
                self.last_k = pred_k
                return
            if not self.last_d:
                self.last_d = pred_d
                return
            if self.last_k > self.last_d and pred_k < pred_d and self.pos_record['pos'] == 0:
                self.last_k = pred_k
                self.last_d = pred_d
                return 'sell'
            elif self.pos_record['pos'] == -1 and self.last_k < self.last_d and pred_k > pred_d:
                self.last_k = pred_k
                self.last_d = pred_d
                return 'close'
            self.last_k = pred_k
            self.last_d = pred_d
        return None

    def predict_action_KDJ(self, kline):
        k = kline.get('K')
        d = kline.get('D')
        j = kline.get('J')
        if self.side == 'long':
            if self.pos_record['pos'] == 0 and k < 20 and d < 20 and j < 20:
                return 'buy'
            elif self.pos_record['pos'] == 1 and k > 80 and d > 80 and j > 80:
                return 'close'
        elif self.side == 'short':
            if self.pos_record['pos'] == 0 and k > 80 and d > 80 and j > 80:
                return 'sell'
            elif self.pos_record['pos'] == -1 and k < 20 and d < 20 and j < 20:
                return 'close'
        return None

    def predict_action_pred(self, kline):
        y = kline.get('C')
        pred_y = kline.get('predict')
        if self.side == 'long':
            if y < pred_y and self.pos_record['pos'] == 0:
                return 'buy'
            elif self.pos_record['pos'] == 1 and y > pred_y:
                return 'close'
        else:
            if y > pred_y and self.pos_record['pos'] == 0:
                return 'sell'
            elif self.pos_record['pos'] == 0 and y < pred_y:
                return 'close'
        return None

    def predict_action_avg(self, kline):
        pred_s = kline.get('signal')
        if self.side == 'long':
            if not self.last_s:
                self.last_s = pred_s
                return
            if self.pos_record['pos'] == 0 and self.last_s != 1 and pred_s == 1:
                self.last_s = pred_s
                return 'buy'
            elif self.pos_record['pos'] == 1 and self.last_s == 1 and pred_s != 1:
                self.last_s = pred_s
                return 'close'
            self.last_s = pred_s
        else:
            if not self.last_s:
                self.last_s = pred_s
                return
            if self.pos_record['pos'] == 0 and self.last_s != -1 and pred_s == -1:
                self.last_s = pred_s
                return 'sell'
            elif self.pos_record['pos'] == -1 and self.last_s == -1 and pred_s != -1:
                self.last_s = pred_s
                return 'close'
            self.last_s = pred_s
        return None

    def cal_position(self, kline, action):
        """
        用于记录做多时买进或做空时卖出的信息
        """
        if action == 'buy' and self.pos_record['pos'] == 0:
            self.pos_record['price'] = kline.get('C') + self.slippage
            self.pos_record['side'] = action
            self.pos_record['pos'] = 1
            self.pos_record['time'] = kline.get('time')
            self.is_trade = True
            self.num = self.init_asset // self.pos_record['price']
        elif action == 'sell' and self.pos_record['pos'] == 0:
            self.pos_record['price'] = kline.get('C') - self.slippage
            self.pos_record['side'] = action
            self.pos_record['time'] = kline.get('time')
            self.pos_record['pos'] = -1
            self.is_trade = True
            self.num = 2 * self.init_asset // self.pos_record['price']
        elif action == 'close':
            self.update_per_record(kline)

    def update_per_record(self, kline):
        """
        用于平仓时的收益计算以及交易信息的记录
        """
        if self.pos_record['side'] == 'buy':
            ret = self.rr(kline.get('C') - self.pos_record['price'], 2) * self.num
            hold_time = (kline.get('time').timestamp() - self.pos_record['time'].timestamp()) / (3600 * 24)
            self.per_record.append([ret, hold_time, ['buy', self.pos_record['price'], self.pos_record['time']], ['sell', kline.get('C'),
                                                                                                                 kline.get('time')]])
            self.acc_ret += ret
            self.pos_record['pos'] = 0
            self.pos_record['side'] = ''
            self.is_trade = True
        elif self.pos_record['side'] == 'sell':
            ret = self.rr(self.pos_record['price'] - kline.get('C'), 2) * self.num
            hold_time = (kline.get('time').timestamp() - self.pos_record['time'].timestamp()) / (3600 * 24)
            self.per_record.append([ret, hold_time, ['sell', self.pos_record['price'], self.pos_record['time']], ['buy', kline.get('C'),
                                                                                                                  kline.get('time')]])
            self.acc_ret += ret
            self.pos_record['pos'] = 0
            self.pos_record['side'] = ''
            self.is_trade = True

    def stat(self, kline, flag=False):
        """
        用于计算每日的收益
        """
        if flag or (self.pos_record['pos'] != 0 and (kline.get('time').timestamp() - self.pos_record['time'].timestamp()) / (3600 * 24) >=
                    self.hold_days):
            self.update_per_record(kline)
            self.is_trade = True
        self.result['total_days'] = self.result.get('total_days', 0) + 1
        if self.is_trade:
            self.result['trade_days'] = self.result.get('trade_days', 0) + 1
        if self.pos_record['side'] == 'buy':
            ret = (kline.get('C') - self.pos_record['price']) * self.num
        elif self.pos_record['side'] == 'sell':
            ret = (self.pos_record['price'] - kline.get('C')) * self.num
        else:
            ret = 0
        self.day_record[kline.get('time')] = self.acc_ret + ret - self.last_ret
        self.last_ret = ret
        self.acc_ret = 0

    def output(self):
        """
        # 输出统计结果
        """
        ret_list = [self.day_record[key] * self.multiple_factor for key in sorted(self.day_record.keys())]  # 每日回报
        if not ret_list:
            return None
        self.time_record = list(self.day_record.keys())
        self.time_record.insert(0, self.test_data[0].get('time'))
        ret_list.insert(0, self.init_asset)
        # 每日净值回报
        for i in range(1, len(ret_list)):
            ret_list[i] = ret_list[i] + ret_list[i - 1]
        if len(self.per_record) == 0:
            return
        self.result['factor'] = self.factor
        self.result['hold_days'] = self.hold_days
        self.result['side'] = self.side
        self.result['time_record'] = self.time_record
        self.result['per_record'] = self.per_record     # 用来打买卖点位置
        self.result['ret_list'] = ret_list
        self.result['total_ret'] = ret_list[-1] - self.init_asset  # 总回报
        return self.result

    @staticmethod
    def rr(x, n=2):
        return round(x, n)

    def draw_pic_tool(self, key, value):
        """
        用于绘制k线图以及交易信息
        """
        kline_x = []
        kline_y = []
        buy_action_x = []
        buy_action = []
        sell_action_x = []
        sell_action = []
        for kline in value.get('data'):
            kline_x.append(kline.get('time'))
            kline_y.append(kline.get('C'))
        for record in value.get('per_record'):
            if record[2][0] == 'buy':
                buy_action.append(record[2][1])
                buy_action_x.append((record[2][2]))
            else:
                sell_action.append(record[2][1])
                sell_action_x.append((record[2][2]))
            if record[3][0] == 'buy':
                buy_action.append(record[3][1])
                buy_action_x.append((record[3][2]))
            else:
                sell_action.append(record[3][1])
                sell_action_x.append((record[3][2]))
        p = figure(tools="crosshair, pan, wheel_zoom, xwheel_pan, ywheel_pan, box_zoom, reset, undo, redo, save",
                   title=f"{key}_{value.get('total_ret')}", x_axis_label='time', y_axis_label='quote', width=1200,
                   height=400,
                   x_axis_type='datetime')
        p.xaxis.major_label_orientation = math.pi / 2
        p.line(kline_x, kline_y, line_color='black', legend="kline")
        p.circle(buy_action_x, buy_action, legend="buy", fill_color="red", line_color="red", size=6)
        p.circle(sell_action_x, sell_action, legend="sell", fill_color="blue", line_color="blue", size=6)
        return p

    def analysis_drawpic(self, result):
        """
        用于绘制回测结果
        """
        result['data'] = self.test_data
        tmp = f"{self.factor}_{result['side']}_{result['hold_days']}"
        fig = self.draw_pic_tool(tmp, result)
        # output_file(f"result_{tmp}_{ToolDateTime.get_date_string('s')}.html")
        show(column(fig))



if __name__ == '__main__':
    for side in ('long','short'):  # 做多还是做空
        for factor in ('rsi','KD','KDJ','pred','avg'):  # 对5种策略依次进行计算
            data = pd.read_csv('TDKJ.csv', parse_dates=['time'])
            data = data[-365:]  # 取最近一年的数据
            list_data = data.to_dict(orient='records')
            
            # 搜索出收益最大的最大持仓天数
            hold_days = 1
            best_hold_days = -1
            max_total_ret = -10000000
            while hold_days <= 365:
                bt = SingleBackTest({'data': list_data, 'side': side, 'factor': factor, 'hold_days':hold_days})
                result = bt.run()
                if not result:
                    hold_days += 1
                    continue
                if result['total_ret'] > max_total_ret:
                    best_hold_days = hold_days
                    max_total_ret = result['total_ret']
                hold_days += 1

            # 绘制收益最大时的交易信息图
            hold_days = best_hold_days
            bt = SingleBackTest({'data': list_data, 'side': side, 'factor': factor, 'hold_days':hold_days})
            result = bt.run()
            bt.analysis_drawpic(result)

            # 打印回测结果
            per_record = result['per_record']
            result.pop('time_record')
            result.pop('data')
            result.pop('per_record')
            result.pop('ret_list')
            df = pd.DataFrame(result, index=[0])
            print(df)
            print('-------------------这是分割线---------------------')

# %% [markdown]
# # 结果分析
# 上面的结果是用天地科技（文件名：TDKJ.csv）近一年的数据进行的预测，每种策略都有一个输出，包括这种策略的
# 总天数（total_days）、交易天数（trade_days）、使用的策略（factor）、最大持仓天数（hold_days）、
# 做多还是做少（side）、总收益（total_ret），通过比较可知，对于这支股票而言，其最优策略为：
# 
# 根据**KDJ指数**进行决策，选择**做空**的方式，最大持仓天数为**26天**，当**K, D, J均大于80**时开仓，当**K, D, J均小于20**时平仓，
# **最大收益为594502.40元**。
# 
# 利用同样的方法，可以获得其他各支股票的在五种策略下的最大收益（单位：万元），如下表所示：
# 
# |            |   rsi   |   KD   |   KDJ   |   神经网络   | AVG |
# | :------------------: | :---: | :---: | :---: | :---: | :------: |
# |        **东风汽车**        |  39.2  |  2.5  |  **55.0**  |  -39.2  |   76.9   |
# |        **中国石化**        |  **47.6**  |  -60.8  |  23.4  |  42.8  |   -18.4   |
# |        **贵州茅台**        |  48.6  |  35.6  |  17.1  |  \  |   **56.2**   |
# |  **天地科技**  | 57.2  | 17.7  | **59.4**  | 50.0  | 30.1 |
# |  **中国银行**  | -15.0 | -178.9 | **-11.2** | -12.8 | -47.1 |
# 
# 从策略上分析，5支股票中有3支的最高收益都出现在KDJ方法，可见这种策略的效果很好；
# 而有3支股票的最低收益都出现在KD方法，可见这种方式的预测效果较差。
# 
# 从股票上分析，贵州茅台和天地科技这两只股票，无论用哪种方式进行预测，最后都是正收益，
# 可将这两只股票的投资价值很强；而中国银行的股票不论用哪种方法，结果都是赔钱，
# 可见如果没有十足的把握，最好不要投资中国银行的股票。

