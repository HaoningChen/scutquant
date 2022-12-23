from . import account, signal_generator  # 别动这行！


class Executor:
    def __init__(self,
                 generator: dict,  # 生成器
                 acc: dict, cost_buy: float, cost_sell: float, min_cost: int,  # 账户和交易费率
                 risk_degree: float = 0.95, auto_offset: bool = False, offset_freq: int = 1,
                 buy_volume: int = 10, sell_volume: int = 10):  # 额外功能，风险控制和自动平仓
        # todo: 增加信号发射器的可选参数
        """
        :param generator: dict, 包括 'mode' 和其它内容, 为执行器找到合适的信号生成方式
        :param acc: dict, 账户
        :param cost_buy: float, 买入费率
        :param cost_sell: float, 卖出费率
        :param min_cost: 最低交易费用
        :param risk_degree: 风险度
        :param auto_offset: 是否自动平仓
        :param offset_freq: 自动平仓的参数，由label构建方式决定
        :param buy_volume: 每次买入的手数
        :param sell_volume: 每次卖出的手数
        """
        self.mode = generator['mode']

        self.init_cash = acc['cash']
        self.position = acc['position']
        self.available = acc['available']
        self.ben_position = acc['ben_position']
        self.ben_cash = acc['cash']
        self.price = None

        self.user_account = None
        self.benchmark = None
        self.cost_buy = cost_buy
        self.cost_sell = cost_sell
        self.min_cost = min_cost
        self.risk_degree = risk_degree
        self.auto_offset = auto_offset
        self.offset_freq = offset_freq
        self.buy_vol = buy_volume
        self.sell_vol = sell_volume

    def create_account(self):
        self.user_account = account.Account(self.init_cash, self.position, self.available, self.price)
        self.benchmark = account.Account(self.ben_cash, self.ben_position, {}, self.price.copy())

    def init_account(self, data):
        """
        :param data: pd.DataFrame, 索引为[('time', 'code')], 列至少应包括 'price' 和 't', 见 execute() 的注释
        :return:
        """
        data_copy = data.copy().reset_index()
        t0 = data_copy['t'][0]
        code = data_copy[data_copy['t'] == t0]['code']
        price0 = data_copy[data_copy['t'] == t0]['price']
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

    def execute(self, data, verbose=0):
        # todo: 增加simulate模式
        """
        :param data: pd.DataFrame, 包括三列：'predict', 't', 'price' 以及多重索引[('time', 'code')]
        :param verbose: bool, 是否输出交易记录
        :return: self
        """

        def check_names(index=data.index, predict='predict', t='t', price='price'):
            names = index.names
            if names[0] != 'time' or names[1] != 'code':
                raise ValueError("index should be like [('time', 'code')]")
            elif predict not in data.columns:
                raise ValueError("data should include column" + predict)
            elif t not in data.columns:
                raise ValueError("data should include column" + t)
            elif price not in data.columns:
                raise ValueError("data should include column" + price)

        check_names()
        Executor.init_account(self, data)
        Executor.create_account(self)
        if self.mode == 'generate':
            time = data['t'].unique()
            for t in time:
                order, current_price = signal_generator.generate(signal=data, index=t, time='t', buy_volume=self.buy_vol,
                                                                 sell_volume=self.sell_vol)
                if verbose == 1:
                    print(order, '\n')
                if self.auto_offset:
                    self.user_account.auto_offset(freq=self.offset_freq, cost_buy=self.cost_buy,
                                                  cost_sell=self.cost_sell, min_cost=self.min_cost)
                trade = self.user_account.check_order(order, current_price)
                self.user_account.update_all(order=order, price=current_price, cost_buy=self.cost_buy,
                                             cost_sell=self.cost_sell, min_cost=self.min_cost, trade=trade)
                self.user_account.risk_control(risk_degree=self.risk_degree, cost_rate=self.cost_sell,
                                               min_cost=self.min_cost)
                self.benchmark.update_all(order=None, price=current_price)
        else:
            raise ValueError('simulate mode is not available by far')
