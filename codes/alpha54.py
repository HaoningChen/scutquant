def ma_n(x, n):
    return x.rolling(window=n).mean()


def ma5(x):
    ma = x.rolling(window=5).mean()
    ma += 1e-10
    return ma


def ma10(x):
    ma = x.rolling(window=10).mean()
    ma += 1e-10
    return ma


def ma20(x):
    ma = x.rolling(window=20).mean()
    return ma


def std_n(x, n):
    return x.rolling(window=n).mean()


def std5(x):
    std = x.rolling(window=5).std()
    return std


def std10(x):
    std = x.rolling(window=10).std()
    return std


def std20(x):
    std = x.rolling(window=20).std()
    return std


def max_n(x, n):
    return x.rolling(window=n).max()


def max5(x):
    m = x.rolling(window=5).max()
    return m


def max10(x):
    m = x.rolling(window=10).max()
    return m


def max20(x):
    m = x.rolling(window=20).max()
    return m


def min_n(x, n):
    return x.rolling(window=n).min()


def min5(x):
    M = x.rolling(window=5).min()
    return M


def min10(x):
    M = x.rolling(window=10).min()
    return M


def min20(x):
    M = x.rolling(window=20).min()
    return M


def beta_n(x, n):
    return ((x.shift(n).fillna(0) + 1e-10) - x) / n


def beta5(x):
    # The rate of price change in the past d periods
    b = ((x.shift(5).fillna(0) + 1e-10) - x) / 5
    return b


def beta10(x):
    b = ((x.shift(10).fillna(0) + 1e-10) - x) / 10
    return b


def beta20(x):
    b = ((x.shift(20).fillna(0) + 1e-10) - x) / 20
    return b


def roc_n(x, n):
    return ((x.shift(n).fillna(0) + 1e-10) - x) / (x + 1e-10)


def roc5(x):
    # rate of change
    r = ((x.shift(5).fillna(0) + 1e-10) - x) / (x + 1e-10)
    return r


def roc10(x):
    r = ((x.shift(10).fillna(0) + 1e-10) - x) / (x + 1e-10)
    return r


def roc20(x):
    r = ((x.shift(20).fillna(0) + 1e-10) - x) / (x + 1e-10)
    return r


def vwap(amount, volume, groupby, n):
    m = amount.groupby([groupby]).rolling(n).sum() / (volume.groupby([groupby]).rolling(n).sum() * 100 + 1e-10)
    return m.sort_index().values


def vwap_series(amount, volume, n):
    m = amount.rolling(n).sum() / (volume.rolling(n).sum() * 100 + 1e-10)
    return m.values


def risk(price, close, groupby, n):
    # 当前一段时间的price和lastClose比较
    r = price.groupby([groupby]).rolling(n).mean() / close.groupby([groupby]).rolling(n).mean() - 1
    return r.sort_index().values


def risk_series(price, close, n):
    # 当前一段时间的price和lastClose比较
    r = price.rolling(n).mean() / close.rolling(n).mean() - 1
    return r.sort_index().values


def hml(high, low, groupby, n):
    # high minus low
    h = high.groupby([groupby]).rolling(n).mean() - low.groupby([groupby]).rolling(n).mean()
    return h.sort_index().values


def hml_series(high, low, n):
    # high minus low
    h = high.rolling(n).mean() - low.rolling(n).mean()
    return h.values


def rsv(price, high, low, groupby, n):
    # Represent the price position between upper and lower resistent price for past n periods
    h = high.groupby([groupby]).rolling(n).max().values
    l = low.groupby([groupby]).rolling(n).min().values
    r = price / (0.5 * (h + l)) - 1
    return r.sort_index().values


def rsv_series(price, high, low, n):
    # Represent the price position between upper and lower resistent price for past n periods
    h = high.rolling(n).max().values
    l = low.rolling(n).min().values
    r = price / (0.5 * (h + l)) - 1
    return r.values


def make_factors(kwargs):
    """
    Only for panel data, you can use other functions instead to make factors for your series data

    :param kwargs:
    {
        data: input data, pd.DataFrame like,
        price: last price or close price, depend on you,
        last_close: close price of last period, or other price you like,
        volume: volume of the current tick,
        amount: amount of the current tick,
        high: the highest price of the current tick,
        low: the lowest price of the current tick,
        label : return rate or other target,
        shift: how many periods the return rate looks forward,
        groupby: groupby,
    }
    :return: pd.DataFrame
    """
    import pandas as pd
    df = kwargs['data']
    price = kwargs['price']
    last_close = kwargs['last_close']
    volume = kwargs['volume']
    amount = kwargs['amount']
    high = kwargs['high']
    low = kwargs['low']
    label = kwargs['label']
    shift = kwargs['shift']
    groupby = kwargs['groupby']
    X = pd.DataFrame()
    if label is not None:
        X['ma5'] = df[label].shift(shift).groupby([groupby]).apply(ma5)
        X['ma10'] = df[label].shift(shift).groupby([groupby]).apply(ma10)
        X['ma20'] = df[label].shift(shift).groupby([groupby]).apply(ma20)

        X['std5'] = df[label].shift(shift).groupby([groupby]).apply(std5)
        X['std10'] = df[label].shift(shift).groupby([groupby]).apply(std10)
        X['std20'] = df[label].shift(shift).groupby([groupby]).apply(std20)

        X['max5'] = df[label].shift(shift).groupby([groupby]).apply(max5)
        X['max10'] = df[label].shift(shift).groupby([groupby]).apply(max10)
        X['max20'] = df[label].shift(shift).groupby([groupby]).apply(max20)

        X['min5'] = df[label].shift(shift).groupby([groupby]).apply(min5)
        X['min10'] = df[label].shift(shift).groupby([groupby]).apply(min10)
        X['min20'] = df[label].shift(shift).groupby([groupby]).apply(min20)

        X['beta5'] = df[label].shift(shift).groupby([groupby]).apply(beta5)
        X['beta10'] = df[label].shift(shift).groupby([groupby]).apply(beta10)
        X['beta20'] = df[label].shift(shift).groupby([groupby]).apply(beta20)

        X['roc5'] = df[label].shift(shift).groupby([groupby]).apply(roc5)
        X['roc10'] = df[label].shift(shift).groupby([groupby]).apply(roc10)
        X['roc20'] = df[label].shift(shift).groupby([groupby]).apply(roc20)
    if price is not None:
        X['MA5'] = df[price].groupby([groupby]).apply(ma5)
        X['MA10'] = df[price].groupby([groupby]).apply(ma10)
        X['MA20'] = df[price].groupby([groupby]).apply(ma20)

        X['STD5'] = df[price].groupby([groupby]).apply(std5)
        X['STD10'] = df[price].groupby([groupby]).apply(std10)
        X['STD20'] = df[price].groupby([groupby]).apply(std20)

        X['MAX5'] = df[price].groupby([groupby]).apply(max5)
        X['MAX10'] = df[price].groupby([groupby]).apply(max10)
        X['MAX20'] = df[price].groupby([groupby]).apply(max20)

        X['MIN5'] = df[price].groupby([groupby]).apply(min5)
        X['MIN10'] = df[price].groupby([groupby]).apply(min10)
        X['MIN20'] = df[price].groupby([groupby]).apply(min20)

        X['BETA5'] = df[price].groupby([groupby]).apply(beta5)
        X['BETA10'] = df[price].groupby([groupby]).apply(beta10)
        X['BETA20'] = df[price].groupby([groupby]).apply(beta20)

        X['ROC5'] = df[price].groupby([groupby]).apply(roc5)
        X['ROC10'] = df[price].groupby([groupby]).apply(roc10)
        X['ROC20'] = df[price].groupby([groupby]).apply(roc20)
    if volume is not None:
        X['vma5'] = df[volume].groupby([groupby]).apply(ma5)
        X['vma10'] = df[volume].groupby([groupby]).apply(ma10)
        X['vma20'] = df[volume].groupby([groupby]).apply(ma20)

        X['vstd5'] = df[volume].groupby([groupby]).apply(std5)
        X['vstd10'] = df[volume].groupby([groupby]).apply(std10)
        X['vstd20'] = df[volume].groupby([groupby]).apply(std20)
    if (volume is not None) and (amount is not None):
        X['vwap5'] = vwap(df[amount], df[volume], groupby=groupby, n=5)
        X['vwap10'] = vwap(df[amount], df[volume], groupby=groupby, n=10)
        X['vwap20'] = vwap(df[amount], df[volume], groupby=groupby, n=20)
    if (last_close is not None) and (price is not None):
        X['risk5'] = risk(df[price], df[last_close], groupby=groupby, n=5)
        X['risk10'] = risk(df[price], df[last_close], groupby=groupby, n=10)
        X['risk20'] = risk(df[price], df[last_close], groupby=groupby, n=20)
    if (high is not None) and (low is not None):
        X['hml5'] = hml(df[high], df[low], groupby=groupby, n=5)
        X['hml10'] = hml(df[high], df[low], groupby=groupby, n=10)
        X['hml20'] = hml(df[high], df[low], groupby=groupby, n=20)
    if (price is not None) and (high is not None) and (low is not None):
        X['rsv5'] = rsv(df[price], df[high], df[low], groupby=groupby, n=5)
        X['rsv10'] = rsv(df[price], df[high], df[low], groupby=groupby, n=10)
        X['rsv20'] = rsv(df[price], df[high], df[low], groupby=groupby, n=20)
    return X


def make_factors_series(kwargs):
    import pandas as pd
    df = kwargs['data']
    price = kwargs['price']
    last_close = kwargs['last_close']
    volume = kwargs['volume']
    amount = kwargs['amount']
    high = kwargs['high']
    low = kwargs['low']
    label = kwargs['label']
    shift = kwargs['shift']
    X = pd.DataFrame()
    if label is not None:
        X['ma5'] = ma5(df[label].shift(shift))
        X['ma10'] = ma10(df[label].shift(shift))
        X['ma20'] = ma20(df[label].shift(shift))

        X['std5'] = std5(df[label].shift(shift))
        X['std10'] = std10(df[label].shift(shift))
        X['std20'] = std20(df[label].shift(shift))

        X['max5'] = max5(df[label].shift(shift))
        X['max10'] = max10(df[label].shift(shift))
        X['max20'] = max20(df[label].shift(shift))

        X['min5'] = min5(df[label].shift(shift))
        X['min10'] = min10(df[label].shift(shift))
        X['min20'] = min20(df[label].shift(shift))

        X['beta5'] = beta5(df[label].shift(shift))
        X['beta10'] = beta10(df[label].shift(shift))
        X['beta20'] = beta20(df[label].shift(shift))

        X['roc5'] = roc5(df[label].shift(shift))
        X['roc10'] = roc10(df[label].shift(shift))
        X['roc20'] = roc20(df[label].shift(shift))
    if price is not None:
        X['MA5'] = ma5(df[price])
        X['MA10'] = ma10(df[price])
        X['MA20'] = ma20(df[price])

        X['STD5'] = std5(df[price])
        X['STD10'] = std10(df[price])
        X['STD20'] = std20(df[price])

        X['MAX5'] = max5(df[price])
        X['MAX10'] = max10(df[price])
        X['MAX20'] = max20(df[price])

        X['MIN5'] = min5(df[price])
        X['MIN10'] = min10(df[price])
        X['MIN20'] = min20(df[price])

        X['BETA5'] = beta5(df[price])
        X['BETA10'] = beta10(df[price])
        X['BETA20'] = beta20(df[price])

        X['ROC5'] = roc5(df[price])
        X['ROC10'] = roc10(df[price])
        X['ROC20'] = roc20(df[price])
    if volume is not None:
        X['vma5'] = ma5(df[volume])
        X['vma10'] = ma10(df[volume])
        X['vma20'] = ma20(df[volume])

        X['vstd5'] = std5(df[volume])
        X['vstd10'] = std10(df[volume])
        X['vstd20'] = std20(df[volume])
    if (volume is not None) and (amount is not None):
        X['vwap5'] = vwap_series(df[amount], df[volume], n=5)
        X['vwap10'] = vwap_series(df[amount], df[volume], n=10)
        X['vwap20'] = vwap_series(df[amount], df[volume], n=20)
    if (last_close is not None) and (price is not None):
        X['risk5'] = risk_series(df[price], df[last_close], n=5)
        X['risk10'] = risk_series(df[price], df[last_close], n=10)
        X['risk20'] = risk_series(df[price], df[last_close], n=20)
    if (high is not None) and (low is not None):
        X['hml5'] = hml_series(df[high], df[low], n=5)
        X['hml10'] = hml_series(df[high], df[low], n=10)
        X['hml20'] = hml_series(df[high], df[low], n=20)
    if (price is not None) and (high is not None) and (low is not None):
        X['rsv5'] = rsv_series(df[price], df[high], df[low], n=5)
        X['rsv10'] = rsv_series(df[price], df[high], df[low], n=10)
        X['rsv20'] = rsv_series(df[price], df[high], df[low], n=20)
    return X
