import pandas as pd


def update_x(x1, x2, n, time='time'):
    # 先入先出栈，容纳指定tick数量的面板数据
    x = pd.concat([x1, x2], axis=0)
    tx = x[time].unique()
    get_predict = False
    if len(tx) >= n:
        t_pop = tx[0]
        x = x[x[time] != t_pop]
        get_predict = True
    return x, get_predict


def update_factors(x, f_kwargs):
    # 更新生成因子的kwargs
    f_kwargs['data'] = x
    return f_kwargs


def buy(code, volume=10, unit='lot'):
    if unit == 'lot':
        volume *= 100
    kwargs = dict.fromkeys(code, volume)
    return kwargs


def sell(code, volume=10, unit='lot'):
    if unit == 'lot':
        volume *= 100
    kwargs = dict.fromkeys(code, volume)
    return kwargs


def simulate(x, index, get_predict, factor_kwargs, xmean, xstd, ymean, ystd, model, price='price', time='time',
             index_level='code', buy_=0.0005, sell_=-0.0005, buy_volume=10, sell_volume=10, unit='lot'):
    """
    :param x: 用于构建因子的数据，滚动更新（输入的是更新过的数据）
    :param index: 当前时间
    :param get_predict: 根据当前x的长度决定是否构建因子并进行预测
    :param factor_kwargs: 构建因子的kwargs
    :param xmean: 用于对因子标准化的x_mean
    :param xstd: 用于对因子标准化的x_std
    :param ymean: 用于还原预测值的y_mean
    :param ystd: 用于还原预测值的y_std
    :param model: 用于预测的模型
    :param price: 用于更新current_price的列的名字
    :param time: 用于筛选当前tick的predict, 从而避免预测值包括不止一个时间段的问题
    :param index_level: 股票代码（或其他标的资产的名字）所在的index的名字
    :param buy_: 触发买入的门槛
    :param sell_: 触发卖出的门槛
    :param buy_volume: 买入数量
    :param sell_volume: 卖出数量
    :param unit: 单位，在中国是按手（'lot'），美国是按股（'share'）
    :return: order字典和current_price字典
    """
    from . import alpha
    buy_list = []
    sell_list = []
    current_price = x.droplevel(0)[price].to_dict()
    if get_predict:
        factor_kwargs = update_factors(x, factor_kwargs)
        # print(factor_kwargs['data'])
        factors = alpha.make_factors(factor_kwargs)
        factors -= xmean
        factors /= xstd
        factors.clip(-5, 5, inplace=True)
        factors = factors.fillna(method='ffill').dropna(axis=0)

        predict = model.predict(factors)
        predict = pd.DataFrame(predict, index=factors.index, columns=['predict'])
        predict += ymean
        predict *= ystd
        predict['t'] = x[time]
        predict = predict[predict['t'] == index]
    else:
        predict = pd.DataFrame()
    if len(predict) > 0:
        # print(predict.index)
        pred_b = predict[predict['predict'].values >= buy_]
        pred_s = predict[predict['predict'].values <= sell_]
        if len(pred_b) > 0:
            buy_list += [i for i in pred_b.index.get_level_values(index_level).to_list()]
        if len(pred_s) > 0:
            sell_list += [i for i in pred_s.index.get_level_values(index_level).to_list()]
    buy_dict = buy(code=buy_list, volume=buy_volume, unit=unit)
    sell_dict = sell(code=sell_list, volume=sell_volume, unit=unit)
    order = {
        'buy': buy_dict,
        'sell': sell_dict
    }
    return order, current_price


def generate(signal, index, pred='predict', time='time', price='price', index_level='code', buy_=0.0005, sell_=-0.0005,
             buy_volume=10, sell_volume=10, unit='lot', buy_only=False):
    """
    :param signal: prediction, pd.DataFrame,  ！！！ SHOULD INCLUDE 2(or 3) COLUMNS：predict, (time), price,
                                                                  AND MULTIINDEX[('time', 'code')]！！！

    :param index: 在循环中的中间变量
    :param time: col index which represents of time, str
    :param price: col index which represents of price, str
    :param pred: str
    :param index_level: str
    :param buy_: min expected return rate for a buy decision, float
    :param sell_: min expected return rate for a sell decision, float
    :param buy_volume: number of lots each time you buy, int
    :param sell_volume: number of lots each time you buy, int
    :param unit: unit of buy or sell, 'lot' or 'share', str
    :param buy_only: bool, in Chinese market, there are strict short-selling restrictions.
    :return: order, dict； current_price(of all stocks available), dict
    """
    # example
    # for idx in IDX:
    #    order, current_price = signal_generator.generate(signal=predict, index=idx, buy_=0.0005, sell_=-0.0005)

    buy_list = []
    sell_list = []
    sig = signal[signal[time] == index]

    sig_b = sig[sig[pred].values >= buy_]
    if len(sig_b) > 0:
        buy_list += [i for i in sig_b.index.get_level_values(index_level).to_list()]
    sig_s = sig[sig[pred].values <= sell_]
    if len(sig_s) > 0:
        sell_list += [i for i in sig_s.index.get_level_values(index_level).to_list()]
    current_price = sig.droplevel(0)[price].to_dict()

    buy_dict = buy(code=buy_list, volume=buy_volume, unit=unit)
    if buy_only:
        sell_dict = {}
    else:
        sell_dict = sell(code=sell_list, volume=sell_volume, unit=unit)
    order = {
        'buy': buy_dict,
        'sell': sell_dict
    }
    return order, current_price


def generate_series(signal, index, pred='predict', price='price', code='code', buy_=0.0005, sell_=-0.0005,
                    buy_volume=10, sell_volume=10, unit='lot'):
    """
    :param signal: prediction, pd.DataFrame
    :param index: 在循环中的中间变量
    :param pred: col name of prediction, str
    :param price: col name of price, str
    :param code: stock code
    :param buy_: float
    :param sell_: float
    :param buy_volume: int
    :param sell_volume: int
    :param unit: str
    :return: order, current price
    """
    """
    example:
    df = df[0:15]
    df['predict'] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1]

    for i in range(len(df)):
        order, price = signal_generator.generate_series(df, index=i, pred='predict', code='000001.SZ')
        print(order)
    """
    buy_list = []
    sell_list = []
    if signal[pred][index] >= buy_:
        buy_list.append(code)
    elif signal[pred][index] <= sell_:
        sell_list.append(code)

    buy_dict = buy(code=buy_list, volume=buy_volume, unit=unit)
    sell_dict = sell(code=sell_list, volume=sell_volume, unit=unit)
    order = {
        'buy': buy_dict,
        'sell': sell_dict
    }
    current_price = {code: signal[price][index]}
    return order, current_price
