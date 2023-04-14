class Account:
    """
    传入：
    init_cash: float
    position: dict {'code': volume}
    available: dict {'code': volume}
    init_price: dict {'code': price}
    order: dict {
                    'buy':{
                            'code': volume,
                        },
                    'sell':{
                            'code': volume,
                        }
                    }
    price: dict {'code': price}
    cost_rate: float, 交易手续费率
    min_cost: int, 最低交易费用
    freq: int (交易后多少个tick自动平仓)
    risk_degree: float, 最大风险度
    """

    def __init__(self, init_cash: float, position: dict, available: dict, init_price: dict):
        self.cash = init_cash  # 可用资金
        self.cash_available = init_cash
        self.position = position  # keys应包括所有资产，如无头寸则值为0，以便按照keys更新持仓
        self.available = available  # 需要持有投资组合的底仓，否则按照T+1制度无法做空
        self.price = init_price  # 资产价格
        self.value = init_cash  # 市值，包括cash和持仓市值
        self.cost = 0.0  # 交易费用
        self.val_hist = []  # 用于绘制市值曲线
        self.buy_hist = []  # 买入记录
        self.sell_hist = []  # 卖出记录
        self.risk = None
        self.risk_curve = []
        # 换手率等于基金在某一时期内的交易额除以该时期内基金的平均市值，再乘以100 %
        self.turnover = []
        self.trade_value = 0.0

    def check_order(self, order, price, cost_rate=0.0015, min_cost=5):  # 检查是否有足够的资金完成order, 如果不够则不买
        # todo: 增加风险度判断（执行该order会不会超出最大风险度）
        cash_inflow = 0.0
        cash_outflow = 0.0
        for code in order['sell'].keys():
            if code in self.available.keys():  # 如果做空的品种有底仓, 则清空底仓
                order["sell"][code] = self.available[code]
            cash_inflow += price[code] * order['sell'][code]
        for code in order['buy'].keys():
            cash_outflow += price[code] * order['buy'][code]
        cost = max(min_cost, (cash_inflow + cash_outflow) * cost_rate)
        cash_needed = cash_outflow - cash_inflow + cost
        if cash_needed > self.cash:
            return order, False
        else:
            return order, True

    def update_price(self, price):  # 更新市场价格
        for code in price.keys():
            self.price[code] = price[code]

    def update_value(self):  # 更新市值
        value_hold = 0.0
        for code in self.price.keys():  # 更新持仓市值, 如果持有的资产在价格里面, 更新资产价值
            if code in self.position.keys():
                value_hold += self.position[code] * self.price[code]
        self.value = self.cash + value_hold
        self.val_hist.append(self.value)

    def update_trade_hist(self, order):  # 更新交易记录
        if order is not None:
            self.buy_hist.append(order['buy'])
            self.sell_hist.append(order['sell'])
        else:
            self.buy_hist.append({})
            self.sell_hist.append({})

    def buy(self, order_buy, cost_rate=0.0015, min_cost=5):  # 买入函数
        buy_value = 0.0
        for code in order_buy.keys():  # 如果资产池里已经有该资产了，那就可以直接加，没有的话就要在字典里添加键值对
            if code in self.position.keys():
                self.position[code] += order_buy[code]  # 更新持仓
                self.available[code] += order_buy[code]  # 更新可交易头寸
            else:
                self.position[code] = order_buy[code]
                self.available[code] = order_buy[code]
            buy_value += self.price[code] * order_buy[code]
            self.trade_value += buy_value
        cost = max(min_cost, buy_value * cost_rate)
        self.cost += cost
        self.cash -= (buy_value + cost)  # 更新现金

    def sell(self, order_sell, cost_rate=0.0005, min_cost=5):  # 卖出函数
        # 做空时, 如果用底仓做空, 则清空底仓; 若融券做空, 则按照"short_volume"参数决定做空数量
        sell_value = 0.0
        for code in order_sell.keys():
            if code in self.position.keys():
                self.position[code] -= order_sell[code]
                self.available[code] -= order_sell[code]
            else:
                self.position[code] = -order_sell[code]
                self.available[code] = -order_sell[code]
            sell_value += self.price[code] * order_sell[code]
            self.trade_value += sell_value
        cost = max(min_cost, sell_value * cost_rate) if sell_value != 0 else 0
        self.cash += (sell_value - cost)  # 更新现金

    def update_all(self, order, price, cost_buy=0.0015, cost_sell=0.0005, min_cost=5, trade=True):
        # 更新市场价格、交易记录、持仓和可交易数量、交易费用和现金，市值
        # order的Key不一定要包括所有资产，但必须是position的子集
        Account.update_price(self, price)  # 首先更新市场价格
        self.trade_value = 0.0
        value_before_trade = self.value
        if order is not None:
            if trade:
                Account.update_trade_hist(self, order)  # 然后更新交易记录
                Account.sell(self, order['sell'], cost_buy, min_cost)
                Account.buy(self, order['buy'], cost_sell, min_cost)
        self.trade_value = abs(self.trade_value)
        self.turnover.append(self.trade_value * 2 / (self.value + value_before_trade))
        Account.update_value(self)

    def auto_offset(self, freq, cost_buy=0.0015, cost_sell=0.0005, min_cost=5):  # 自动平仓
        """
        example: 对某只股票的买入记录为[4, 1, 1, 2, 3], 假设买入后2个tick平仓, 则自动平仓应为[nan, nan, 4, 1, 1]

        :param freq: 多少个tick后平仓, 例如收益率构建方式为close_-2 / close_-1 - 1, delta_t=1, 所以是1tick后平仓
        :param cost_buy: 买入费率
        :param cost_sell: 卖出费率
        :param min_cost: 最小交易费用
        :return:
        """
        if len(self.buy_hist) >= freq:
            offset_buy = self.sell_hist[-freq]
            offset_sell = self.buy_hist[-freq]
            Account.sell(self, offset_sell, cost_sell, min_cost)  # 卖出平仓
            Account.buy(self, offset_buy, cost_buy, min_cost)  # 买入平仓

    def risk_control(self, risk_degree, cost_rate=0.0005, min_cost=5):  # 控制风险, 当风险度超过计划风险度时, 按比例减少持仓
        # 令risk回到risk_degree: 各资产持仓量为向量x, 各资产市场价格为向量p, 总市值为v, 风险度 r = p*x/v. 即px = rv.
        # 求出减持比例b, 使得r = p*(1-b)x/v = risk_degree. 即1-b = risk_degree * v / (p * x), b = 1- (risk_degree * v) / (p*x)
        # 代入px=rv，得b = 1 - (risk_degree * v) / (r * v) = 1 - risk_degree / r
        self.risk = 1 - self.cash / self.value
        self.risk_curve.append(self.risk)
        if self.risk > risk_degree:
            b = 1 - risk_degree / self.risk
            sell_order = self.position.copy()
            for code in sell_order.keys():
                sell_order[code] *= b
                sell_order[code] = int(sell_order[code] / 100 + 0.5) * 100  # 以手为单位减持
            Account.sell(self, sell_order, cost_rate, min_cost)
