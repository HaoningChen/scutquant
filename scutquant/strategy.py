def get_volume(num=10, unit=None):
    """
    生成每笔交易的交易量

    :return: int, 交易量, 单位是允许交易的最小单位
    """
    n = num  # 可以自定义函数, 例如求解最优资产组合等
    if unit == 'lot':
        n *= 100
    return n


def trade(code, num=10, unit=None):
    """
    :param code: code of assets to be traded
    :param num: number of units
    :param unit: 'lot' or else
    :return: dict like {'code': volume}
    """
    vol = get_volume(num, unit)
    kwargs = dict.fromkeys(code, vol)
    return kwargs


def get_assets_list(data, index_level):
    lis = []
    if len(data) > 0:
        lis += [i for i in data.index.get_level_values(index_level).to_list()]
    return lis


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
            kwargs["volume"] = 10
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95

        self.buy = kwargs["buy"]
        self.sell = kwargs["sell"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.buy_only = kwargs["buy_only"]
        self.num = kwargs["volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]

    def to_signal(self, data, pred="predict", index_level="code"):
        """
        :param data: pd.DataFrame, 面板数据，包含单个tick下所有资产的预测值
        :param pred: str, 预测值所在的列
        :param index_level: str, groupby的依据，一般为资产id所在的索引名
        :return: dict
        """
        data_buy = data[data[pred].values >= self.buy]
        data_sell = data[data[pred].values <= self.sell]

        buy_list, sell_list = get_assets_list(data_buy, index_level), get_assets_list(data_sell, index_level)

        buy_dict = trade(buy_list, num=self.num, unit=self.unit)
        if self.buy_only:
            sell_dict = {}
        else:
            sell_dict = trade(code=sell_list, num=self.num, unit=self.unit)
        order = {
            'buy': buy_dict,
            'sell': sell_dict
        }
        return order


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
            kwargs["volume"] = 10
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95

        self.k = kwargs["k"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.buy_only = kwargs["buy_only"]
        self.num = kwargs["volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]

    def to_signal(self, data, pred="predict", index_level="code"):
        n_k = int(len(data) * self.k + 0.5)

        data_ = data.copy().sort_values("predict", ascending=False)
        data_buy = data_.head(n_k)
        data_sell = data_.tail(n_k)

        buy_list, sell_list = get_assets_list(data_buy, index_level), get_assets_list(data_sell, index_level)

        buy_dict = trade(buy_list, num=self.num, unit=self.unit)
        if self.buy_only:
            sell_dict = {}
        else:
            sell_dict = trade(code=sell_list, num=self.num, unit=self.unit)
        order = {
            'buy': buy_dict,
            'sell': sell_dict
        }
        return order
