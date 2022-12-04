from . import account, signal_generator


class Executor:
    def __init__(self,
                 generator: dict,  # 生成器
                 acc: dict, cost_buy: float, cost_sell: float, min_cost: int,  # 账户和交易费率
                 risk_degree: float = 0.95, auto_offset: bool = False, offset_freq: int = 1):  # 额外功能，风险控制和自动平仓
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
        """
        self.mode = generator['mode']

        self.init_cash = acc['cash']
        self.position = acc['position']
        self.available = acc['available']
        self.price = acc['price']

        self.user_account = None
        self.benchmark = None
        self.cost_buy = cost_buy
        self.cost_sell = cost_sell
        self.min_cost = min_cost
        self.risk_degree = risk_degree
        self.is_auto_offset = auto_offset
        self.offset_freq = offset_freq

    def create_account(self):
        self.user_account = account.Account(self.init_cash, self.position, self.available, self.price)
        self.benchmark = account.Account(self.init_cash, self.position.copy(), self.available.copy(), self.price.copy())

    def execute(self, data):
        # todo: 增加simulate模式
        """
        :param data: pd.DataFrame, 包括三列：'predict', 'time', 'price' 以及多重索引[('time', 'code')]
        :return: self
        """
        Executor.create_account(self)
        if self.mode == 'generate':
            time = data['time'].unique()
            for t in time:
                order, current_price = signal_generator.generate(signal=data, index=t)
                if self.is_auto_offset:
                    self.user_account.auto_offset(freq=self.offset_freq, cost_buy=self.cost_buy,
                                                  cost_sell=self.cost_sell, min_cost=self.min_cost)
                self.user_account.check_order(order, current_price)
                self.user_account.update_all(order=order, price=current_price, cost_buy=self.cost_buy,
                                             cost_sell=self.cost_sell, min_cost=self.min_cost)
                self.user_account.risk_control(risk_degree=self.risk_degree, cost_rate=self.cost_sell,
                                               min_cost=self.min_cost)
                self.benchmark.update_all(order=None, price=current_price)
        else:
            raise ValueError('simulate mode is not available by far')
