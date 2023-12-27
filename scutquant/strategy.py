"""
所有信息以字典的形式呈现, 例如
price = {
    "000001.SZ": 10,
    ......
}
"""
import pandas as pd


def raw_prediction_to_signal(pred: pd.Series, total_cash: float, long_only: bool = False) -> pd.Series:
    pred -= pred.groupby(level=0).mean()
    pred_ = pred.copy()
    abs_sum = abs(pred_).groupby(level=0).sum()
    pred_ /= abs_sum
    if long_only:
        pred_[pred_ < 0] = 0
        pred_ *= 2
    return pred_ * total_cash


def get_trade_volume(signal: pd.Series, price: pd.Series, volume: pd.Series, threshold: float = 0.05,
                     unit: str = "lot") -> pd.Series:
    """
    :param signal: 中性化后的组合份额 * 总资金
    :param price: 股票价格
    :param volume: 股票成交量
    :param threshold:
    :param unit:
    :return:
    """
    trade_volume = abs(signal) / price  # 得到单位是股
    multiplier = 100 if unit == "lot" else 1
    trade_volume /= multiplier
    max_volume = volume * threshold
    trade_volume_ = trade_volume.copy()
    trade_volume_[trade_volume_ > max_volume] = max_volume
    return (trade_volume_ + 0.5).astype(int) * multiplier  # 四舍五入取整, 最后以股作为单位


def get_price(data: pd.DataFrame, price: str = "price") -> dict:
    current_price = data.droplevel(0)[price].to_dict()
    return current_price


def get_vol(data: pd.DataFrame, volume: str = "volume") -> dict:
    current_volume = data.droplevel(0)[volume].to_dict()
    return current_volume


def check_signal(order: dict) -> dict:
    return {k: v for k, v in order.items() if v > 0}


class BaseStrategy:
    def __init__(self, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if "k" not in kwargs.keys():
            kwargs["k"] = 0.2
        if "auto_offset" not in kwargs.keys():
            kwargs["auto_offset"] = False
        if "offset_freq" not in kwargs.keys():
            kwargs["offset_freq"] = 1
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05
        if "long_only" not in kwargs.keys():
            kwargs["long_only"] = False

        self.k = kwargs["k"]
        self.auto_offset = kwargs["auto_offset"]
        self.offset_freq = kwargs["offset_freq"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]
        self.long_only = kwargs["long_only"]

    def to_signal(self, **kwargs):
        """
        对输入的每个tick的价格和预测值，输出买和卖的信息
        注意这里是先买后卖, 但执行的时候是先卖后买

        :return: dict like {'buy': {'code': volume}, 'sell': {'code': volume}}
        """
        pass


class TopKStrategy(BaseStrategy):
    """
    受分组累计收益率启发(参考report模块的group_return_ana): 做多预测收益率最高的n个资产, 做空预测收益率最低的n个资产
    这里的k是个百分数, 意为做多资产池中排前k%的资产, 做空排后k%的资产
    """

    def to_signal(self, data: pd.DataFrame, cash_available: float = None, **kwargs):
        """
        根据每天的预测值构造多空组合, 分别买入和卖出预测值的top K和bottom K, 而不管自身的position
        """
        n_k = int(len(data) * self.k + 0.5)
        price = get_price(data, price="price")
        data_ = data.copy().sort_values("predict", ascending=False)
        data_buy = data_.head(n_k)
        data_sell = data_.tail(n_k)

        concat_data = pd.concat([data_buy, data_sell], axis=0)
        concat_data["predict"] = raw_prediction_to_signal(concat_data["predict"], total_cash=cash_available,
                                                          long_only=self.long_only)

        concat_data["trade_volume"] = get_trade_volume(concat_data["predict"], concat_data["price"],
                                                       concat_data["volume"], threshold=self.max_volume,
                                                       unit=self.unit)

        buy_dict = get_vol(concat_data[concat_data.index.isin(data_buy.index)], volume="trade_volume")
        sell_dict = get_vol(concat_data[concat_data.index.isin(data_sell.index)], volume="trade_volume")

        buy_dict, sell_dict = check_signal(buy_dict), check_signal(sell_dict)
        order = {
            "buy": buy_dict,
            "sell": sell_dict
        }
        return order, price


class LongStrategy(BaseStrategy):
    """
    每天持有top K组的多头, 清空不在当日signal里的仓位, 买入或调整在signal的股票
    """

    def __init__(self, kwargs=None):
        super().__init__(kwargs)
        if kwargs is None:
            kwargs = {}
        if "k" not in kwargs.keys():
            kwargs["k"] = 0.2
        if "unit" not in kwargs.keys():
            kwargs["unit"] = "lot"
        if "risk_degree" not in kwargs.keys():
            kwargs["risk_degree"] = 0.95
        if "max_volume" not in kwargs.keys():
            kwargs["max_volume"] = 0.05

        self.k = kwargs["k"]
        self.auto_offset = False
        self.offset_freq = 0
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]
        self.long_only = True

    def to_signal(self, data: pd.DataFrame, position: dict, cash_available: float = None):
        n_k = int(len(data) * self.k + 0.5)
        price = get_price(data, price="price")
        data_buy = data.copy().sort_values("predict", ascending=False).head(n_k)
        data_buy["predict"] = raw_prediction_to_signal(data_buy["predict"], total_cash=cash_available,
                                                       long_only=self.long_only)
        # print(data_buy["predict"].groupby(level=0).sum())
        data_buy["trade_volume"] = get_trade_volume(data_buy["predict"], data_buy["price"], data_buy["volume"],
                                                    threshold=self.max_volume, unit=self.unit)
        position = check_signal(position)
        buy_dict = get_vol(data_buy, volume="trade_volume")
        buy_dict1 = {k: v for k, v in buy_dict.items() if k not in position.keys()}
        buy_dict2 = {k: v - position[k] for k, v in buy_dict.items() if k in position.keys() and v >= position[k]}
        sell_dict1 = {k: position[k] - v for k, v in buy_dict.items() if k in position.keys() and v < position[k]}
        sell_dict2 = {k: v for k, v in position.items() if k not in buy_dict.keys()}

        buy_dict1.update(buy_dict2)
        sell_dict1.update(sell_dict2)
        buy_dict, sell_dict = check_signal(buy_dict1), check_signal(sell_dict1)
        order = {
            "buy": buy_dict,
            "sell": sell_dict
        }
        return order, price
