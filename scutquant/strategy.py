"""
所有信息以字典的形式呈现, 例如
price = {
    "000001.SZ": 10,
    ......
}
"""
import pandas as pd


def raw_prediction_to_signal(pred: pd.Series, total_cash: float, long_only: bool = False) -> pd.Series:
    """
    构造组合并乘可投资资金, 得到每只股票分配的资金
    """
    if len(pred) > 1:
        if not long_only:
            pred -= pred.groupby(level=0).mean()
        pred_ = pred.copy()
        abs_sum = abs(pred_).groupby(level=0).sum()
        pred_ /= abs_sum
        return pred_ * total_cash
    else:
        return pred * total_cash


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
    """
    只返回value>0的键值对
    """
    return {k: v for k, v in order.items() if v > 0}


class BaseStrategy:
    def __init__(self, kwargs=None):
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
        if "long_only" not in kwargs.keys():
            kwargs["long_only"] = False

        self.k = kwargs["k"]
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


class QlibTopKStrategy(BaseStrategy):
    """
    第一天持有整个valid broad top k 组股票, 以后每天卖出持仓的bottom k组股票, 买入持仓外对应数量的股票
    即假设valid broad=n, 第一天买入n*k支股票, 以后每天卖出n*k^2支股票, 买入n*k^2支股票
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
        if "equal_weight" not in kwargs.keys():
            kwargs["equal_weight"] = True

        self.n_start = kwargs["n_start"] if "n_start" in kwargs.keys() else None
        self.k = kwargs["k"]
        self.unit = kwargs["unit"]
        self.risk_degree = kwargs["risk_degree"]
        self.max_volume = kwargs["max_volume"]
        self.equal_weight = kwargs["equal_weight"]
        self.long_only = True

    def to_signal(self, data: pd.DataFrame, position: dict, cash_available: float = None):
        n_k = int(len(data) * self.k + 0.5)
        valid_position = check_signal(position)
        price = get_price(data, price="price")
        if len(valid_position) == 0:  # 如果当前没有仓位, 买入valid broad top k组
            if self.n_start is not None:
                n_k = self.n_start
            data_buy = data.copy().sort_values("predict", ascending=False).head(n_k)  # 买入的股票
            sell_dict = {}
        else:
            swap_k = int(len(valid_position) * self.k + 0.5)

            data_in_position = data[data.index.get_level_values(1).isin(valid_position.keys())]
            data_sell = data_in_position.copy().sort_values("predict", ascending=False).tail(swap_k)
            sell_dict = {k: v for k, v in valid_position.items() if k in data_sell.index.get_level_values(1)}

            data_not_in_position = data[~data.index.get_level_values(1).isin(valid_position.keys())]
            data_buy = data_not_in_position.copy().sort_values("predict", ascending=False).head(swap_k)

        if self.equal_weight:
            data_buy["predict"] = cash_available / n_k  # 等权持有, 乘上总金额
        else:
            data_buy["predict"] = raw_prediction_to_signal(data_buy["predict"], total_cash=cash_available,
                                                           long_only=True)  # 加权持有
        data_buy["trade_volume"] = get_trade_volume(data_buy["predict"], data_buy["price"], data_buy["volume"],
                                                    threshold=self.max_volume, unit=self.unit)
        buy_dict = check_signal(get_vol(data_buy, volume="trade_volume"))
        sell_dict = check_signal(sell_dict)

        order = {
            "buy": buy_dict,
            "sell": sell_dict
        }
        return order, price
