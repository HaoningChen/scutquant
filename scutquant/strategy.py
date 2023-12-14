"""
所有信息以字典的形式呈现, 例如
price = {
    "000001.SZ": 10,
    ......
}
"""


def get_price(data, price: str = "price") -> dict:
    current_price = data.droplevel(0)[price].to_dict()
    return current_price


def get_vol(data, volume: str = "volume") -> dict:
    current_volume = data.droplevel(0)[volume].to_dict()
    return current_volume


def check_signal(order: dict) -> dict:
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
        if "long_only" not in kwargs.keys():
            kwargs["long_only"] = False
        if "short_volume" not in kwargs.keys():
            kwargs["short_volume"] = 0  # 为0时，只能用底仓做空; >0时允许融券做空
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05

        self.k = kwargs["k"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.long_only = kwargs["long_only"]
        self.num = kwargs["short_volume"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]

    def to_signal(self, data, pred="predict", index_level="code"):
        n_k = int(len(data) * self.k + 0.5)
        price = get_price(data, price="price")
        data_ = data.copy().sort_values("predict", ascending=False)
        data_buy = data_.head(n_k)
        data_sell = data_.tail(n_k)

        buy_dict = get_vol(data_buy, volume="trade_volume")
        if self.long_only:
            sell_dict = {}
        else:
            sell_dict = get_vol(data_sell, volume="trade_volume")

        buy_dict, sell_dict = check_signal(buy_dict), check_signal(sell_dict)
        order = {
            "buy": buy_dict,
            "sell": sell_dict
        }
        return order, price
