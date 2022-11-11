def ma(x, n):
    return x.rolling(window=n).mean()


def std(x, n):
    return x.rolling(window=n).mean()


def max_(x, n):
    return x.rolling(window=n).max()


def min_(x, n):
    return x.rolling(window=n).min()


def beta(x, n):
    # The rate of price change in the past d periods
    return ((x.shift(n).fillna(0) + 1e-10) - x) / n


def roc(x, n):
    # rate of change
    return ((x.shift(n).fillna(0) + 1e-10) - x) / (x + 1e-10)


def skew(x, n):
    s = x.rolling(n).skew()
    return s


def kurt(x, n):
    k = x.rolling(n).kurt()
    return k


def qtlu(x, n):
    # The 80% quantile of past d periods' price
    q = x.rolling(n).quantile(0.8)
    return q.sort_index().values


def qtld(x, n):
    # The 20% quantile of past d periods' price
    q = x.rolling(n).quantile(0.2)
    return q.sort_index().values


def vwap(amount, volume, groupby, n):
    m = amount.groupby([groupby]).rolling(n).sum() / (volume.groupby([groupby]).rolling(n).sum() * 100 + 1e-10)
    m = m.sort_index()
    return m.values


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
    return r.values


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
        X['ma5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: ma(x, 5))
        X['ma10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: ma(x, 10))
        X['ma20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: ma(x, 20))

        X['std5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: std(x, 5))
        X['std10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: std(x, 10))
        X['std20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: std(x, 20))

        X['max5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: max_(x, 5))
        X['max10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: max_(x, 10))
        X['max20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: max_(x, 20))

        X['min5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: min_(x, 5))
        X['min10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: min_(x, 10))
        X['min20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: min_(x, 20))

        # X['skew5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: skew(x, 5))
        # X['skew10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: skew(x, 10))
        # X['skew20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: skew(x, 20))

        # X['kurt5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: kurt(x, 5))
        # X['kurt10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: kurt(x, 10))
        # X['kurt20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: kurt(x, 20))

        X['beta5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: beta(x, 5))
        X['beta10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: beta(x, 10))
        X['beta20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: beta(x, 20))

        X['roc5'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: roc(x, 5))
        X['roc10'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: roc(x, 10))
        X['roc20'] = df[label].groupby([groupby]).shift(shift).groupby([groupby]).apply(lambda x: roc(x, 20))
    if price is not None:
        X['MA5'] = df[price].groupby([groupby]).apply(lambda x: ma(x, 5))
        X['MA10'] = df[price].groupby([groupby]).apply(lambda x: ma(x, 10))
        X['MA20'] = df[price].groupby([groupby]).apply(lambda x: ma(x, 20))

        X['STD5'] = df[price].groupby([groupby]).apply(lambda x: std(x, 5))
        X['STD10'] = df[price].groupby([groupby]).apply(lambda x: std(x, 10))
        X['STD20'] = df[price].groupby([groupby]).apply(lambda x: std(x, 20))

        X['MAX5'] = df[price].groupby([groupby]).apply(lambda x: max_(x, 5))
        X['MAX10'] = df[price].groupby([groupby]).apply(lambda x: max_(x, 10))
        X['MAX20'] = df[price].groupby([groupby]).apply(lambda x: max_(x, 20))

        X['MIN5'] = df[price].groupby([groupby]).apply(lambda x: min_(x, 5))
        X['MIN10'] = df[price].groupby([groupby]).apply(lambda x: min_(x, 10))
        X['MIN20'] = df[price].groupby([groupby]).apply(lambda x: max_(x, 20))

        # X['SKEW5'] = df[price].groupby([groupby]).apply(lambda x: skew(x, 5))
        # X['SKEW10'] = df[price].groupby([groupby]).apply(lambda x: skew(x, 10))
        # X['SKEW20'] = df[price].groupby([groupby]).apply(lambda x: skew(x, 20))

        # X['KURT5'] = df[price].groupby([groupby]).apply(lambda x: kurt(x, 5))
        # X['KURT10'] = df[price].groupby([groupby]).apply(lambda x: kurt(x, 10))
        # X['KURT20'] = df[price].groupby([groupby]).apply(lambda x: kurt(x, 20))

        X['BETA5'] = df[price].groupby([groupby]).apply(lambda x: beta(x, 5))
        X['BETA10'] = df[price].groupby([groupby]).apply(lambda x: beta(x, 10))
        X['BETA20'] = df[price].groupby([groupby]).apply(lambda x: beta(x, 20))

        X['ROC5'] = df[price].groupby([groupby]).apply(lambda x: roc(x, 5))
        X['ROC10'] = df[price].groupby([groupby]).apply(lambda x: roc(x, 10))
        X['ROC20'] = df[price].groupby([groupby]).apply(lambda x: roc(x, 20))
    if volume is not None:
        X['vma5'] = df[volume].groupby([groupby]).apply(lambda x: ma(x, 5))
        X['vma10'] = df[volume].groupby([groupby]).apply(lambda x: ma(x, 10))
        X['vma20'] = df[volume].groupby([groupby]).apply(lambda x: ma(x, 20))

        X['vstd5'] = df[volume].groupby([groupby]).apply(lambda x: std(x, 5))
        X['vstd10'] = df[volume].groupby([groupby]).apply(lambda x: std(x, 5))
        X['vstd20'] = df[volume].groupby([groupby]).apply(lambda x: std(x, 5))
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
        X['ma5'] = ma(df[label].shift(shift), 5)
        X['ma10'] = ma(df[label].shift(shift), 10)
        X['ma20'] = ma(df[label].shift(shift), 20)

        X['std5'] = std(df[label].shift(shift), 5)
        X['std10'] = std(df[label].shift(shift), 10)
        X['std20'] = std(df[label].shift(shift), 20)

        X['max5'] = max_(df[label].shift(shift), 5)
        X['max10'] = max_(df[label].shift(shift), 10)
        X['max20'] = max_(df[label].shift(shift), 20)

        X['min5'] = min_(df[label].shift(shift), 5)
        X['min10'] = min_(df[label].shift(shift), 10)
        X['min20'] = min_(df[label].shift(shift), 20)

        # X['skew5'] = skew(df[label].shift(shift), 5)
        # X['skew10'] = skew(df[label].shift(shift), 10)
        # X['skew20'] = skew(df[label].shift(shift), 20)

        # X['kurt5'] = kurt(df[label].shift(shift), 5)
        # X['kurt10'] = kurt(df[label].shift(shift), 10)
        # X['kurt20'] = kurt(df[label].shift(shift), 20)

        X['beta5'] = beta(df[label].shift(shift), 5)
        X['beta10'] = beta(df[label].shift(shift), 10)
        X['beta20'] = beta(df[label].shift(shift), 20)

        X['roc5'] = roc(df[label].shift(shift), 5)
        X['roc10'] = roc(df[label].shift(shift), 10)
        X['roc20'] = roc(df[label].shift(shift), 20)
    if price is not None:
        X['MA5'] = ma(df[price], 5)
        X['MA10'] = ma(df[price], 10)
        X['MA20'] = ma(df[price], 20)

        X['STD5'] = std(df[price], 5)
        X['STD10'] = std(df[price], 10)
        X['STD20'] = std(df[price], 20)

        X['MAX5'] = max_(df[price], 5)
        X['MAX10'] = max_(df[price], 10)
        X['MAX20'] = max_(df[price], 20)

        X['MIN5'] = min_(df[price], 5)
        X['MIN10'] = min_(df[price], 10)
        X['MIN20'] = min_(df[price], 20)

        # X['SKEW5'] = skew(df[price], 5)
        # X['SKEW10'] = skew(df[price], 10)
        # X['SKEW20'] = skew(df[price], 20)

        # X['KURT5'] = kurt(df[price], 5)
        # X['KURT10'] = kurt(df[price], 10)
        # X['KURT20'] = kurt(df[price], 20)

        X['BETA5'] = beta(df[price], 5)
        X['BETA10'] = beta(df[price], 10)
        X['BETA20'] = beta(df[price], 20)

        X['ROC5'] = roc(df[price], 5)
        X['ROC10'] = roc(df[price], 10)
        X['ROC20'] = roc(df[price], 20)
    if volume is not None:
        X['vma5'] = ma(df[volume], 5)
        X['vma10'] = ma(df[volume], 10)
        X['vma20'] = ma(df[volume], 20)

        X['vstd5'] = std(df[volume], 5)
        X['vstd10'] = std(df[volume], 10)
        X['vstd20'] = std(df[volume], 20)
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
