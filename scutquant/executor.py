from . import account, strategy
from .signal_generator import *


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
        self.value_hold = 0.0
        for code in self.user_account.price.keys():  # 更新持仓市值, 如果持有的资产在价格里面, 更新资产价值
            if code in self.user_account.position.keys():
                self.value_hold += self.user_account.position[code] * self.user_account.price[code]
        # value是账户总价值, 乘risk_deg后得到所有可交易资金, 减去value_hold就是剩余可交易资金
        return self.user_account.value * self.s.risk_degree - self.value_hold

    def execute(self, data: pd.DataFrame, verbose: int = 0):
        """
        :param data: pd.DataFrame, 包括三列：'predict', 'volume', 'price', 'label' 以及多重索引[('time', 'code')]
        :param verbose: int, 是否输出交易记录
        :return: self
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
            time = data.index.get_level_values(0).unique().values
            for t in time:
                idx = data["R"].groupby(data.index.names[0]).mean()  # 大盘收益率
                self.time.append(t)
                data_select = data[data.index.get_level_values(0) == t]
                signal = generate(data=data_select, strategy=self.s, cash_available=self.get_cash_available())
                order, current_price = signal["order"], signal["current_price"]

                if self.s.auto_offset:
                    order = self.user_account.generate_total_order(order=order, freq=self.s.offset_freq)
                order, trade = self.user_account.check_order(order, current_price)

                if verbose == 1 and trade:
                    print(t, '\n', "buy:", '\n', order["buy"], '\n', "sell:", order["sell"], '\n')

                self.user_account.update_all(order=order, price=current_price, cost_buy=self.cost_buy,
                                             cost_sell=self.cost_sell, min_cost=self.min_cost, trade=trade)
                self.user_account.risk_control(risk_degree=self.s.risk_degree, cost_rate=self.cost_sell,
                                               min_cost=self.min_cost)
                self.benchmark.value *= (1 + idx[idx.index == t][0])  # 乘上1+大盘收益率, 相当于等权指数
                self.benchmark.val_hist.append(self.benchmark.value)
        else:
            raise ValueError("simulate mode is not available by far")
