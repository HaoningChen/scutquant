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


def update_factors(x: pd.DataFrame, f_kwargs: dict) -> dict:
    # 更新生成因子的kwargs
    f_kwargs["data"] = x
    return f_kwargs


def simulate(x: pd.DataFrame, current_time, get_predict: bool, factor_kwargs: dict, ymean, ystd, model, strategy,
             cash_available: float = None, price: str = "price", volume: str = "volume"):
    """
    # 用于模拟实盘, 即动态更新因子和预测值

    :param x: 用于构建因子的数据，滚动更新（输入的是更新过的数据）, 需要额外增加price(shift(-1))和volume(shift(-1))
    :param current_time: 当前时间
    :param get_predict: 根据当前x的长度决定是否构建因子并进行预测
    :param factor_kwargs: 构建因子的kwargs
    :param ymean: 用于还原预测值的y_mean
    :param ystd: 用于还原预测值的y_std
    :param model: 用于预测的模型
    :param strategy: 用于生成指令的策略
    :param cash_available: 生成指令时，账户的可用资金
    :param price: 用于更新成交价格(一般为shift(-1))
    :param volume: 成交量(一般为shift(-1))
    :return: order字典和current_price字典
    """
    from . import alpha
    if get_predict:
        factor_kwargs = update_factors(x, factor_kwargs)
        factors = alpha.make_factors(factor_kwargs)
        factors -= x.groupby(x.index.names[1]).mean()
        factors /= x.groupby(x.index.names[1]).std()
        factors.clip(-3, 3, inplace=True)
        factors = factors.fillna(method="ffill").dropna(axis=0)

        predict = model.predict(factors)
        predict = pd.DataFrame(predict, index=factors.index, columns=["predict"])
        predict["predict"] += ymean  # 考虑换成截面上的均值的均值
        predict["predict"] *= ystd  # 考虑换成截面上的标准差的均值
        predict = predict[predict.index.get_level_values(0) == current_time]
        predict["price"] = x[x.index.get_level_values(0) == current_time][price]
        predict["volume"] = x[x.index.get_level_values(0) == current_time][volume]
    else:
        predict = pd.DataFrame()
    if len(predict) > 0:
        order, current_price = strategy.to_signal(predict, cash_available=cash_available)
    else:
        order = None
        current_price = None
    dic = {
        "order": order,
        "current_price": current_price,
    }
    return dic


def generate(data: pd.DataFrame, strategy, cash_available: float = None) -> dict:
    """
    :param data: prediction, pd.DataFrame
    :param strategy: 策略
    :param cash_available: 可用于投资的资金
    :return: order, dict； current_price(of all stocks available), dict
    """

    order, current_price = strategy.to_signal(data, cash_available=cash_available)
    dic = {
        "order": order,
        "current_price": current_price,
    }
    return dic
