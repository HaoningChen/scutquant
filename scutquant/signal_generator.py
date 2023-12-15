import pandas as pd


def update_x(x1: pd.DataFrame, x2: pd.DataFrame, n: int) -> tuple[pd.DataFrame, bool]:
    # 先入先出栈，容纳指定tick数量的面板数据
    x = pd.concat([x1, x2], axis=0)
    tx = x.index.get_level_values(0).unique()
    get_predict = False
    if len(tx) >= n:  # 到达容量后, pop掉时间最前的tick
        t_pop = tx[0]
        x = x[x.index.get_level_values(0) != t_pop]
        get_predict = True
    return x, get_predict


def prepare(predict: pd.DataFrame, data: pd.DataFrame, price: str, volume: str, real_ret: pd.Series) -> pd.DataFrame:
    """
    :param predict: pd.DataFrame, 预测值, 应包括"predict"
    :param data: pd.DataFrame, 提供时间和价格信息
    :param price: str, data中表示价格的列名
    :param volume: str, data中表示成交量的列名
    :param real_ret: pd.Series, 真实收益率
    :return: pd.DataFrame
    """
    data_ = data.copy()
    predict.columns = ["predict"]
    index = predict.index
    data1 = data_[data_.index.isin(index)]
    data1 = data1.reset_index()
    data1 = data1.set_index(predict.index.names).sort_index()
    predict["price"] = data1[price]
    predict["volume"] = data1[volume]  # 当天的交易量, 假设交易量不会发生大的跳跃
    predict.index.names = ["time", "code"]
    predict["price"] = predict["price"].groupby(["code"]).shift(-1)  # 指令是T时生成的, 但是T+1执行, 所以是shift(-1)
    predict["R"] = real_ret[real_ret.index.isin(predict.index)]  # 本来就是T+2对T+1的收益率, 因此不用前移
    return predict.dropna()


def raw_prediction_to_signal(pred: pd.Series, total_cash: float, long_only: bool = False) -> pd.Series:
    pred -= pred.groupby(level=0).mean()
    if long_only:
        pred[pred.values < 0] = 0
        abs_sum = pred[pred.values > 0].groupby(level=0).sum()
    else:
        abs_sum = abs(pred).groupby(level=0).sum()
    pred /= abs_sum
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
    trade_volume = abs(signal) / price  # 现在是以股作为单位
    if unit == "lot":
        trade_volume /= 100  # 除以100使其变成以手为单位
    max_volume = volume * threshold  # 这也是以手为单位
    trade_volume.where(trade_volume <= max_volume, max_volume)
    return (trade_volume + 0.5).astype(int) * 100  # 四舍五入取整, 最后以股作为单位


def generate(data: pd.DataFrame, strategy, cash_available: float = None, is_raw: bool = True,
             threshold: float = 0.05, unit: str = "lot") -> dict:
    """
    :param data: 包括预测值, 交易价格和交易量的pd.DataFrame
    :param strategy: 策略
    :param cash_available: 可用于投资的资金
    :param is_raw: 输入的data是否为raw prediction
    :param threshold: 最大交易占比
    :param unit: 交易单位
    :return: order, dict； current_price(of all stocks available), dict
    """
    if is_raw:
        data["predict"] = raw_prediction_to_signal(data["predict"], cash_available, long_only=strategy.long_only)
    data["trade_volume"] = get_trade_volume(data["predict"], data["price"], data["volume"],
                                            threshold=threshold, unit=unit)
    order, current_price = strategy.to_signal(data)
    dic = {
        "order": order,
        "current_price": current_price,
    }
    return dic
