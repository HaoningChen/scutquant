from . import account, strategy
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def get_daily_inter(data: pd.Series | pd.DataFrame, shuffle=False):
    daily_count = data.groupby(level=0).size().values
    daily_index = np.roll(np.cumsum(daily_count), 1)
    daily_index[0] = 0
    if shuffle:
        daily_shuffle = list(zip(daily_index, daily_count))
        np.random.shuffle(daily_shuffle)
        daily_index, daily_count = zip(*daily_shuffle)
    return daily_index, daily_count


def prepare(predict: pd.DataFrame, data: pd.DataFrame, price: str, volume: str, real_ret: pd.Series) -> pd.DataFrame:
    """
    :param predict: pd.DataFrame, 预测值, 应包括"predict"
    :param data: pd.DataFrame, 提供时间和价格信息
    :param price: str, data中表示价格的列名
    :param volume: str, data中表示成交量的列名
    :param real_ret: pd.Series, 真实收益率
    :return: pd.DataFrame
    """
    data_ = data.copy()
    predict.columns = ["predict"]
    index = predict.index
    data1 = data_[data_.index.isin(index)]
    data1 = data1.reset_index()
    data1 = data1.set_index(predict.index.names).sort_index()
    predict["price"] = data1[price]
    predict["volume"] = data1[volume]  # 当天的交易量, 假设交易量不会发生大的跳跃
    predict.index.names = ["time", "code"]
    predict["price"] = predict["price"].groupby(["code"]).shift(-1)  # 指令是T时生成的, 但是T+1执行, 所以是shift(-1)
    predict["R"] = real_ret[real_ret.index.isin(predict.index)]  # 本来就是T+2对T+1的收益率, 因此不用前移
    return predict.dropna()


class Executor:
    def __init__(self, generator: dict, stra, acc: dict, trade_params: dict):
        """
        :param generator: dict, 包括 'mode' 和其它内容, 为执行器找到合适的信号生成方式
        :param acc: dict, 账户

        """
        if acc is None:
            acc = {}
        keys = acc.keys()
        if "cash" not in keys:
            acc["cash"] = 1e8
        if "position" not in keys:
            acc["position"] = {}
        if "available" not in keys:
            acc["available"] = {}

        self.mode = generator['mode']

        self.init_cash: float = acc['cash']
        self.position: dict = acc['position']
        self.value_hold: float = 0.0
        self.available: dict = acc['available']
        self.ben_cash: float = acc['cash']

        self.price = None
        self.time = []

        self.user_account = None
        self.benchmark = None
        self.cost_buy = trade_params["cost_buy"]
        self.cost_sell = trade_params["cost_sell"]
        self.min_cost = trade_params["min_cost"]

        self.s = getattr(strategy, stra["class"])
        self.s = self.s(stra["kwargs"])

    def init_account(self, data: pd.DataFrame):
        """
        :param data: pd.DataFrame, 索引为[('time', 'code')], 列至少应包括 'price' 和 't', 见 execute() 的注释
        :return:
        """
        data_copy = data.copy()
        t0 = data_copy.index.get_level_values(0)[0]
        code = data_copy[data_copy.index.get_level_values(0) == t0].index.get_level_values(1).values
        price0 = data_copy[data_copy.index.get_level_values(0) == t0]['price']
        price_zip = zip(code, price0)
        self.price = dict(price_zip)
        if self.position is None:  # 如果没有position自然也没有available, 将它们初始化为0
            zero_list = [0 for _ in range(len(code))]
            position_zip, available_zip = zip(code, zero_list), zip(code, zero_list)
            self.position = dict(position_zip)
            self.available = dict(available_zip)

    def create_account(self):
        self.user_account = account.Account(self.init_cash, self.position, self.available, self.price)
        self.benchmark = account.Account(self.ben_cash, {}, {}, self.price.copy())

    def get_cash_available(self):
        """
        fixme: 调整计算方式使其适应先卖后买的情况
        之所以不用cash * risk_degree是因为先卖后买的情况下, 当前的cash跟实际可支配的cash不一样(因为卖了就有钱了)
        """
        return self.user_account.value * self.s.risk_degree

    def execute(self, data: pd.DataFrame, verbose: int = 0):
        """
        :param data: pd.DataFrame, 包括三列：'predict', 'volume', 'price', 'label' 以及多重索引[('time', 'code')]
        :param verbose: int, 是否输出交易记录
        :return:
        """

        def check_names(index=data.index, predict="predict", price="price"):
            names = index.names
            if names[0] != "time" or names[1] != "code":
                raise ValueError("index should be like [('time', 'code')]")
            elif predict not in data.columns:
                raise ValueError("data should include column" + predict)
            elif price not in data.columns:
                raise ValueError("data should include column" + price)

        check_names()
        self.init_account(data)
        self.create_account()
        if self.mode == "generate":
            benchmark = data["R"].groupby(level=0).transform(lambda x: x.mean())  # 大盘收益率
            daily_idx, daily_count = get_daily_inter(data)
            for idx, count in zip(daily_idx, daily_count):
                batch = slice(idx, idx + count)
                data_batch = data.iloc[batch]
                benchmark_batch = benchmark.iloc[batch]
                current_day = data_batch.index.get_level_values(0)[0]
                self.time.append(current_day)
                order, current_price = self.s.to_signal(data_batch, position=self.user_account.position,
                                                        cash_available=self.get_cash_available())

                if self.s.auto_offset:
                    order = self.user_account.adjust_order(order=order, freq=self.s.offset_freq)
                order = self.user_account.check_order(order, current_price, risk=self.s.risk_degree)
                # print(trade)
                # trade = True
                if verbose == 1:
                    print(current_day, '\n', "buy:", '\n', order["buy"], '\n', "sell:", order["sell"], '\n')

                self.user_account.update_all(order=order, price=current_price, cost_buy=self.cost_buy,
                                             cost_sell=self.cost_sell, min_cost=self.min_cost)
                self.user_account.risk_control(risk_degree=self.s.risk_degree, cost_rate=self.cost_sell,
                                               min_cost=self.min_cost)

                self.benchmark.value *= 1 + benchmark_batch.values[0]  # 乘上1+大盘收益率, 相当于等权指数
                self.benchmark.val_hist.append(self.benchmark.value)
        else:
            raise ValueError("simulate mode is not available by far")
