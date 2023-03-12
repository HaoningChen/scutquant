def get_volume(asset, price=None, volume=None, cash_available=None, num=10, unit=None, max_volume=0.002):
    """
    生成每笔交易的交易量
    :param asset: list
    :param price: dict {"code": price}
    :param volume: dict {"code": volume}
    :param cash_available: float
    :param num: int, default volume
    :param unit: string, "lot" or None
    :param max_volume: float, 最大交易手数是volume的max_volume倍
    :return: int, 交易量, 单位是允许交易的最小单位
    """
    max_vol = {}
    if volume is not None:
        for k in volume.keys():
            max_vol[k] = int(volume[k] * max_volume + 0.5)
    if (price is None) or (cash_available is None):
        n = [num for _ in range(len(asset))]  # 可以自定义函数, 例如求解最优资产组合等
        if unit == 'lot':
            for i in range(len(n)):
                n[i] = n[i] * 100
    else:
        invest = cash_available / len(asset) if len(asset) != 0 else 0
        n = []
        if unit == 'lot':
            for a in asset:
                # 考虑到大单不一定成交，最大不超过max_volume手
                n.append(int(invest / (price[a] * 100)) if int(invest / (price[a] * 100)) <= max_vol[a]
                         else max_vol[a])
            for i in range(len(n)):
                n[i] = n[i] * 100
        else:
            for a in asset:
                # 考虑到大单不一定成交，最大不超过max_volume手, 即max_volume * 100股
                n.append(int(invest / price[a]) if int(invest / price[a]) <= max_vol[a] else max_vol[a])
    return n


def trade(code, num=10, unit=None, price=None, volume=None, cash_available=None, max_volume=0.002):
    """
    :param code: code of assets to be traded
    :param num: number of units
    :param unit: 'lot' or else
    :param price: dict, 资产价格
    :param volume: 资产的可交易数量
    :param cash_available: float or int
    :param max_volume: float, 最大交易手数是volume的max_volume倍
    :return: dict like {'code': volume}
    """
    vol = get_volume(asset=code, price=price, volume=volume, cash_available=cash_available, num=num, unit=unit,
                     max_volume=max_volume)
    kwargs = dict(zip(code, vol))
    return kwargs


def get_assets_list(data, index_level):
    lis = []
    if len(data) > 0:
        lis += [i for i in data.index.get_level_values(index_level).to_list()]
    return lis


def get_price(data, price="price"):
    current_price = data.droplevel(0)[price].to_dict()
    return current_price


def get_vol(data, volume="volume"):
    current_volume = data.droplevel(0)[volume].to_dict()
    return current_volume


def check_signal(order):
    order_ = order.copy()
    for k in order_.keys():
        if order[k] <= 0:
            order.pop(k)
    return order


class BaseStrategy:
    def __init__(self, **kwargs):
        pass

    def to_signal(self, **kwargs):
        """
        对输入的每个tick的价格和预测值，输出买和卖的信息

        :return: dict like {'buy': {'code': volume}, 'sell': {'code': volume}}
        """
        pass


class BaselineStrategy(BaseStrategy):
    """
    非常简单的低买高卖: 预测未来的收益率大于某个数时, 买入, 并在未来平仓; 预测未来的收益率小于某个负数时, 做空, 同样在未来平仓
    在中国市场, 默认以手为单位, 所以volume要乘100; 同时由于严格的做空限制, 我们可以设buy_only=True以禁止做空, 但可以做多然后平仓
    """

    def __init__(self, kwargs=None):
        super().__init__()
        if kwargs is None:
            kwargs = {}
        if "buy" not in kwargs.keys():
            kwargs["buy"] = 0.005  # 预测收益率超过0.5%则买入. 如果模型是二分类模型，把它设为1即可. kwargs["sell"]同理(设为-1)
        if "sell" not in kwargs.keys():
            kwargs["sell"] = -0.005
        if "auto_offset" not in kwargs.keys():
            kwargs["auto_offset"] = False
        if "offset_freq" not in kwargs.keys():
            kwargs["offset_freq"] = 1
        if "buy_only" not in kwargs.keys():
            kwargs["buy_only"] = False
        if "trade_volume" not in kwargs.keys():
            kwargs["volume"] = 50
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05

        self.buy = kwargs["buy"]
        self.sell = kwargs["sell"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.buy_only = kwargs["buy_only"]
        self.num = kwargs["volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]

    def to_signal(self, data, pred="predict", index_level="code", cash_available=None):
        """
        :param data: pd.DataFrame, 面板数据，包含单个tick下所有资产的预测值
        :param pred: str, 预测值所在的列
        :param index_level: str, groupby的依据，一般为资产id所在的索引名
        :param cash_available: float or int
        :return: dict
        """
        data_buy = data[data[pred].values >= self.buy]
        data_sell = data[data[pred].values <= self.sell]
        price = get_price(data, "price")
        volume = get_vol(data, "volume")

        buy_list, sell_list = get_assets_list(data_buy, index_level), get_assets_list(data_sell, index_level)

        buy_dict = trade(code=buy_list, num=self.num, unit=self.unit, price=price, volume=volume,
                         cash_available=cash_available, max_volume=self.max_volume)
        if self.buy_only:
            sell_dict = {}
        else:
            sell_dict = trade(code=sell_list, num=self.num, unit=self.unit, price=price, volume=volume,
                              cash_available=None, max_volume=self.max_volume)

        buy_dict, sell_dict = check_signal(buy_dict), check_signal(sell_dict)
        order = {
            'buy': buy_dict,
            'sell': sell_dict
        }
        return order, price


class TopKStrategy(BaseStrategy):
    """
    受分组累计收益率启发(参考report模块的group_return_ana): 做多预测收益率最高的n个资产, 做空预测收益率最低的n个资产,
                                                    并在未来平仓. 此时得到的收益率是最高的.
    同样地, 由于中国市场难以做空, 我们仍然可以设buy_only=True. 这样得到的收益就是Group1的收益.
    这里的k是个百分数, 意为做多资产池中排前k%的资产, 做空排后k%的资产
    """

    def __init__(self, kwargs=None):
        super().__init__()
        if kwargs is None:
            kwargs = {}
        if "k" not in kwargs.keys():
            kwargs["k"] = 0.2
        if "auto_offset" not in kwargs.keys():
            kwargs["auto_offset"] = False
        if "offset_freq" not in kwargs.keys():
            kwargs["offset_freq"] = 1
        if "buy_only" not in kwargs.keys():
            kwargs["buy_only"] = False
        if "trade_volume" not in kwargs.keys():
            kwargs["volume"] = 50
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05

        self.k = kwargs["k"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.buy_only = kwargs["buy_only"]
        self.num = kwargs["volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]

    def to_signal(self, data, pred="predict", index_level="code", cash_available=None):
        n_k = int(len(data) * self.k + 0.5)

        data_ = data.copy().sort_values("predict", ascending=False)
        data_buy = data_.head(n_k)
        data_sell = data_.tail(n_k)
        price = get_price(data, "price")
        volume = get_vol(data, "volume")

        buy_list, sell_list = get_assets_list(data_buy, index_level), get_assets_list(data_sell, index_level)

        buy_dict = trade(buy_list, num=self.num, unit=self.unit, price=price, volume=volume,
                         cash_available=cash_available, max_volume=self.max_volume)
        if self.buy_only:
            sell_dict = {}
        else:
            sell_dict = trade(code=sell_list, num=self.num, unit=self.unit, price=price, volume=volume,
                              cash_available=None, max_volume=self.max_volume)

        buy_dict, sell_dict = check_signal(buy_dict), check_signal(sell_dict)
        order = {
            'buy': buy_dict,
            'sell': sell_dict
        }
        return order, price


class StrictTopKStrategy(BaseStrategy):
    """
    做多Group1中收益率大于buy的股票，做空Group5中收益率小于sell的股票
    """

    def __init__(self, kwargs=None):
        super().__init__()
        if "k" not in kwargs.keys():
            kwargs["k"] = 0.2
        if "buy" not in kwargs.keys():
            kwargs["buy"] = 0.005  # 预测收益率超过0.5%则买入. 如果模型是二分类模型，把它设为1即可. kwargs["sell"]同理(设为-1)
        if "sell" not in kwargs.keys():
            kwargs["sell"] = -0.005
        if "auto_offset" not in kwargs.keys():
            kwargs["auto_offset"] = False
        if "offset_freq" not in kwargs.keys():
            kwargs["offset_freq"] = 1
        if "buy_only" not in kwargs.keys():
            kwargs["buy_only"] = False
        if "trade_volume" not in kwargs.keys():
            kwargs["volume"] = 50
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05

        self.k = kwargs["k"]
        self.buy = kwargs["buy"]
        self.sell = kwargs["sell"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.buy_only = kwargs["buy_only"]
        self.num = kwargs["volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]

    def to_signal(self, data, pred="predict", index_level="code", cash_available=None):
        price = get_price(data, "price")
        volume = get_vol(data, "volume")
        n_k = int(len(data) * self.k + 0.5)
        data_ = data.copy().sort_values("predict", ascending=False)

        data_buy = data_.head(n_k)
        data_buy = data_buy[data_buy[pred] >= self.buy]
        buy_list = get_assets_list(data_buy, index_level)
        buy_dict = trade(buy_list, num=self.num, unit=self.unit, price=price, volume=volume,
                         cash_available=cash_available, max_volume=self.max_volume)

        sell_dict = {}
        if not self.buy_only:
            data_sell = data_.tail(n_k)
            data_sell = data_sell[data_sell[pred] <= self.sell]
            sell_list = get_assets_list(data_sell, index_level)
            sell_dict = trade(sell_list, num=self.num, unit=self.unit, price=price, volume=volume, cash_available=None,
                              max_volume=self.max_volume)

        buy_dict, sell_dict = check_signal(buy_dict), check_signal(sell_dict)
        order = {
            'buy': buy_dict,
            'sell': sell_dict
        }
        return order, price


class SigmaStrategy(BaseStrategy):
    def __init__(self, kwargs=None):
        super().__init__()
        keys = kwargs.keys()
        if "sigma" not in keys:
            kwargs["sigma"] = 2
        if "buy" not in keys:
            kwargs["buy"] = 0
        if "sell" not in keys:
            kwargs["sell"] = -0
        if "auto_offset" not in kwargs.keys():
            kwargs["auto_offset"] = False
        if "offset_freq" not in kwargs.keys():
            kwargs["offset_freq"] = 1
        if "buy_only" not in kwargs.keys():
            kwargs["buy_only"] = False
        if "trade_volume" not in kwargs.keys():
            kwargs["volume"] = 50
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05

        self.sigma = kwargs["sigma"]
        self.buy = kwargs["buy"]
        self.sell = kwargs["sell"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.buy_only = kwargs["buy_only"]
        self.num = kwargs["volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]

    def to_signal(self, data, pred="predict", index_level="code", cash_available=None):
        price = get_price(data, "price")
        volume = get_vol(data, "volume")
        mean = data[pred].values.mean()
        std = data[pred].values.std()

        buy_ = self.sigma * std + mean
        data_buy = data[data[pred] >= buy_]
        data_buy = data_buy[data_buy[pred] >= self.buy]
        buy_list = get_assets_list(data_buy, index_level)
        buy_dict = trade(buy_list, num=self.num, unit=self.unit, price=price, volume=volume,
                         cash_available=cash_available, max_volume=self.max_volume)

        sell_dict = {}
        if not self.buy_only:
            sell_ = mean - self.sigma * std
            data_sell = data[data[pred] <= sell_]
            data_sell = data_sell[data_sell[pred] <= self.sell]
            sell_list = get_assets_list(data_sell, index_level)
            sell_dict = trade(sell_list, num=self.num, unit=self.unit, price=price, volume=volume, cash_available=None,
                              max_volume=self.max_volume)
        buy_dict, sell_dict = check_signal(buy_dict), check_signal(sell_dict)
        order = {
            'buy': buy_dict,
            'sell': sell_dict
        }
        return order, price
