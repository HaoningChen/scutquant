from . import account, signal_generator, strategy  # 别动这行！


def prepare(predict, data, price, volume):
    """
    :param predict: pd.DataFrame, 预测值, 应包括"predict"
    :param data: pd.DataFrame, 提供时间和价格信息
    :param price: str, data中表示价格的列名
    :param volume: str, data中表示成交量的列名
    :return: pd.DataFrame
    """
    data_ = data.copy()
    predict.columns = ['predict']
    index = predict.index
    data1 = data_[data_.index.isin(index)]
    data1 = data1.reset_index()
    data1 = data1.set_index(predict.index.names).sort_index()
    predict['price'] = data1[price]
    predict['volume'] = data1[volume]
    predict.index.names = ['time', 'code']
    predict["price"] = predict["price"].groupby(["code"]).shift(-1)  # 指令是T时生成的, 但是T+1执行, 所以是shift(-1)
    return predict.dropna()


class Executor:
    def __init__(self, generator, stra, acc, trade_params):
        # todo: 增加信号发射器的可选参数
        """
        :param generator: dict, 包括 'mode' 和其它内容, 为执行器找到合适的信号生成方式
        :param acc: dict, 账户

        """
        if acc is None:
            acc = {}
        keys = acc.keys()
        if "cash" not in keys:
            acc["cash"] = 1e9
        if "position" not in keys:
            acc["position"] = None
        if "available" not in keys:
            acc["available"] = None
        if "ben_position" not in keys:
            acc["ben_position"] = None

        self.mode = generator['mode']

        self.init_cash = acc['cash']
        self.position = acc['position']
        self.value_hold = 0.0
        self.available = acc['available']
        self.ben_position = acc['ben_position']
        self.ben_cash = acc['cash']

        self.price = None
        self.time = []

        self.user_account = None
        self.benchmark = None
        self.cost_buy = trade_params["cost_buy"]
        self.cost_sell = trade_params["cost_sell"]
        self.min_cost = trade_params["min_cost"]

        self.s = getattr(strategy, stra["class"])
        self.s = self.s(stra["kwargs"])

    def init_account(self, data):
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
        if self.ben_position is None:
            cash_invest = self.init_cash / len(code)  # 将可用资金均匀地投资每项资产(虽然这样做是不对的，应该分配不同权重), 得到指数
            self.ben_position = dict()  # 不用初始化available，因为不交易
            for code in self.price.keys():
                self.ben_position[code] = int(cash_invest / (self.price[code] * 100) + 0.5) * 100  # 四舍五入取整, 以百为单位
                self.ben_cash -= self.ben_position[code] * self.price[code]

    def create_account(self):
        self.user_account = account.Account(self.init_cash, self.position, self.available, self.price)
        self.benchmark = account.Account(self.ben_cash, self.ben_position, {}, self.price.copy())

    def get_cash_available(self):
        self.value_hold = 0.0
        for code in self.user_account.price.keys():  # 更新持仓市值, 如果持有的资产在价格里面, 更新资产价值
            if code in self.user_account.position.keys():
                self.value_hold += self.user_account.position[code] * self.price[code]
        return self.user_account.value * self.s.risk_degree - self.value_hold

    def execute(self, data, verbose=0):
        # todo: 增加simulate模式
        """
        :param data: pd.DataFrame, 包括三列：'predict', 't', 'price' 以及多重索引[('time', 'code')]
        :param verbose: bool, 是否输出交易记录
        :return: self
        """

        def check_names(index=data.index, predict='predict',  price='price'):
            names = index.names
            if names[0] != 'time' or names[1] != 'code':
                raise ValueError("index should be like [('time', 'code')]")
            elif predict not in data.columns:
                raise ValueError("data should include column" + predict)
            elif price not in data.columns:
                raise ValueError("data should include column" + price)

        check_names()
        Executor.init_account(self, data)
        Executor.create_account(self)
        if self.mode == 'generate':
            time = data.index.get_level_values(0).unique().values
            for t in time:
                self.time.append(t)
                data_select = data[data.index.get_level_values(0) == t]
                signal = signal_generator.generate(data=data_select, strategy=self.s,
                                                   cash_available=Executor.get_cash_available(self))
                order, current_price = signal["order"], signal["current_price"]

                if self.s.auto_offset:
                    self.user_account.auto_offset(freq=self.s.offset_freq, cost_buy=self.cost_buy,
                                                  cost_sell=self.cost_sell, min_cost=self.min_cost)
                order, trade = self.user_account.check_order(order, current_price)

                if verbose == 1:
                    print(order, '\n')

                self.user_account.update_all(order=order, price=current_price, cost_buy=self.cost_buy,
                                             cost_sell=self.cost_sell, min_cost=self.min_cost, trade=trade)
                self.user_account.risk_control(risk_degree=self.s.risk_degree, cost_rate=self.cost_sell,
                                               min_cost=self.min_cost)
                self.benchmark.update_all(order=None, price=current_price)
        else:
            raise ValueError('simulate mode is not available by far')
