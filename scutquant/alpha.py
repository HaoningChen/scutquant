"""
因子是用来解释超额收益的变量，分为alpha因子和风险因子(beta因子)。alpha因子旨在找出能解释市场异象的变量，而风险因子是宏观因素(系统性风险)的代理变量，二者并没有严格的区分
此库包含的因子均为量价数据因子，属于技术分析流派，技术分析的前提条件是市场无效（依据有三：一是交易者非理性，二是存在套利限制，三是存在市场异象）

因子构建流程：(1)首先根据表达式生成因子（即本库的功能），然后进行标准化得到因子暴露

            PS:因子构建需要有依据（市场异象/已有的研究/自己发现的规律等，本库基于第二条构建）

            (2)然后计算大类因子之间的相关系数（可以将大类下面的因子做简单平均）并决定是否正交化（当然也有研究指出这没有必要，而且在机器学习模型下这其实不影响结果）
            (3)最后检验因子质量（用pearson corr或者mutual information等指标，对因子和目标值进行回归），剔除得分较低的因子

            PS：构建因子之后的工作也就是第一、二次培训的内容，可以先用alpha.make_factors()生成因子，然后用scutquant.auto_process()完成剩下的流程

因子检验方法（我怎么能够放心地使用我的因子）：
    (1)截面回归，假设因子库（因子的集合）为X，目标值（一般为下期收益率）为y，回归方程 y = alpha + Beta * X，根据t统计量判断beta是否显著不为0.
    该检验是检验单个因子是否显著，缺点是因子与收益率的关系未必是线性的，因此无法检验非线性关系的因子，同时它也没有考虑因子相互作用的情况，即交互项
    (2)自相关性、分层效应、IC分析等: 参考 https://b23.tv/hDB4ZLW
    (3)另一种t检验：已知IC均值ic, ICIR = ic/std(IC), 时间长度为n, 则 t = ic / (ICIR/sqrt(n))，该检验是检验整个因子库是否显著

    PS: ic, ICIR等数据可以用scutquant.ic_ana()获得

因子检验流程（自用）
    (1)构建模型前：参考 因子构建流程 第3步
    (2)构建模型后：计算IC, ICIR, RANK_IC, RANK_ICIR，完成以下工作：
        1、自相关性，旨在得到因子持续性方面的信息（个人理解是因子生命周期长度）
        2、分层效应（即一般书里面的投资组合排序法，但是我们的排序依据是模型的预测值而非因子暴露）
        3、IC序列，IC均值和ICIR，不同频率的IC可视化（月均IC热力图）
        4、t检验（因子检验方法 里面的（3))

因子检验流程（石川等《因子投资方法与实践》）
    (1)排序法
    PS: 由于他们用的模型是线性回归模型，即 y_it = alpha_i + BETA_i * X_it + e_it, 因此对于x_kit in X_it, 其因子收益率就是对应的beta_ki * x_kit.
        其中x_kit为xk因子对于资产i在t时刻的因子暴露，即xk因子标准化后在t时刻的值.
        所有因子经过标准化后，可得到一个纯因子组合，这是因子模拟投资组合的前提条件

        1、将资产池内的资产在截面上按照排序变量（要检验的因子值）进行排序
        2、按照排序结果将资产分为n组（一般取n=5或n=10）
        3、对每组资产计算其收益率，由于投资组合的收益率已经尽可能由目标因子驱动，因此该投资组合的收益率序列就是因子收益率序列（原书p23，存疑）
        4、得到因子收益率后，进行假设检验（H0为收益率=0, H1为收益率>0）, 令r因子收益率的时间序列，则构建t统计量：t = mean(r) / std(r),
           并进行5%水平下的双侧检验（书上p26原话是这么说，但如果预期因子收益率为正数则应该用单侧检验?）
    (2)截面回归（或时序回归），即 因子检验方法 (1)，并做t检验
"""
import pandas as pd


def ma(x, n):
    m = x.rolling(window=n).mean()
    return m.sort_index().values


def std(x, n):
    s = x.rolling(window=n).std()
    return s.sort_index().values


def max_(x, n):
    m = x.rolling(window=n).max()
    return m.sort_index().values


def min_(x, n):
    m = x.rolling(window=n).min()
    return m.sort_index().values


def beta(x, n):
    # The rate of price change in the past d periods
    b = ((x.shift(n).fillna(x.mean()) + 1e-10) - x.shift(0)) / n
    return b.sort_index().values


def roc(x, n):
    # rate of change
    r = ((x.shift(n).fillna(x.mean()) + 1e-10) - x.shift(0)) / (x.shift(0) + 1e-10)
    return r.sort_index().values


def kmid(close, open, groupby, n):
    k = close.groupby([groupby]).rolling(n).mean() / open.groupby([groupby]).rolling(n).mean() - 1
    return k.sort_index().values


def kmid2(close, open, high, low, groupby, n):
    # 作用不大
    k = close.groupby([groupby]).rolling(n).mean() - open.groupby([groupby]).rolling(n).mean()
    hl = high.groupby([groupby]).rolling(n).mean() - low.groupby([groupby]).rolling(n).mean() + 1e-12
    k = k / hl
    return k.sort_index().values


def klen(high, low, open, groupby, n):
    k = high.groupby([groupby]).rolling(n).mean() - low.groupby([groupby]).rolling(n).mean()
    k = k / open.groupby([groupby]).rolling(n).mean()
    return k.sort_index().values


def ksft(close, open, high, low, groupby, n):
    k = 2 * close.groupby([groupby]).rolling(n).mean() - high.groupby([groupby]).rolling(n).mean() - \
        low.groupby([groupby]).rolling(n).mean()
    k = k / open.groupby([groupby]).rolling(n).mean()
    return k.sort_index().values


def ksft2(close, high, low, groupby, n):
    # 作用不大
    k = 2 * close.groupby([groupby]).rolling(n).mean() - high.groupby([groupby]).rolling(n).mean() - \
        low.groupby([groupby]).rolling(n).mean()
    hl = high.groupby([groupby]).rolling(n).mean() - low.groupby([groupby]).rolling(n).mean() + 1e-12
    k = k / hl
    return k.sort_index().values


def vwap(amount, volume, groupby, n):
    # 加权成交均价
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


def make_factors(kwargs=None, windows=None, raw_data=10):
    """
    面板数据适用，序列数据请移步 make_factors_series

    一个例子：
        df = df.set_index(['time', 'code']).sort_index()

        df['ret'] = q.price2ret(price=df['lastPrice'])

        kwargs = {
            'data': df,
            'label': 'ret',
            'price': 'lastPrice',
            'last_close': 'lastClose',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
            'amount': 'amount',
            'groupby': 'code',
            'shift': 21
        }

        X = alpha.make_factors(kwargs)

    :param kwargs:
    {
        data: pd.DataFrame, 输入的数据
        price: str, 最新价格
        open: str, 最近的收盘价（昨天或者其它）
        volume: str, 当前tick的交易量
        amount: str, 当前tick的交易额
        high: str, 当前tick的最高价
        low: str, 当前tick的最低价
        label : str, 目标值
        shift: int, 应滞后的阶数，由label的构建方法决定
        groupby: str, 排序的标准（一般为'code'）
    }
    :param windows: list, 移动窗口的列表
    :param raw_data: 原始数据的滞后项
    :return: pd.DataFrame
    """
    if kwargs is None:
        kwargs = {}
    if "data" not in kwargs.keys():
        kwargs["data"] = pd.DataFrame()
    if "price" not in kwargs.keys():
        kwargs["price"] = "close"
    if "open" not in kwargs.keys():
        kwargs["open"] = "open"
    if "volume" not in kwargs.keys():
        kwargs["volume"] = "volume"
    if "amount" not in kwargs.keys():
        kwargs["amount"] = "amount"
    if "high" not in kwargs.keys():
        kwargs["high"] = "high"
    if "low" not in kwargs.keys():
        kwargs["low"] = "low"
    if "shift" not in kwargs.keys():
        kwargs["shift"] = 1
    if "groupby" not in kwargs.keys():
        kwargs["groupby"] = "code"

    df = kwargs['data']
    price = kwargs['price']
    last_close = kwargs['open']
    volume = kwargs['volume']
    amount = kwargs['amount']
    high = kwargs['high']
    low = kwargs['low']
    label = kwargs['label']
    shift = kwargs['shift']
    groupby = kwargs['groupby']
    if windows is None:
        windows = [5, 10, 20, 30, 60]

    X = pd.DataFrame()
    if label is not None:
        data = df[label].groupby(groupby).shift(shift)
        group = data.groupby([groupby])
        for n in range(raw_data):
            X[label + str(n)] = group.shift(n)
        for w in windows:  # rolling windows
            X['ma' + str(w)] = ma(group, w)
            X['std' + str(w)] = std(group, w)
            X['max' + str(w)] = max_(group, w)
            X['min' + str(w)] = min_(group, w)
            X['beta' + str(w)] = beta(group, w)
            X['roc' + str(w)] = roc(group, w)

    if price is not None:
        group = df[price].groupby([groupby])
        for n in range(raw_data):
            X[price + str(n)] = group.shift(n).sort_index().values
        for w in windows:
            X['MA' + str(w)] = ma(group, w)
            X['STD' + str(w)] = std(group, w)
            X['MAX' + str(w)] = max_(group, w)
            X['MIN' + str(w)] = min_(group, w)
            X['BETA' + str(w)] = beta(group, w)
            X['ROC' + str(w)] = roc(group, w)

    if volume is not None:
        group = df[volume].groupby([groupby])
        for n in range(raw_data):
            X[volume + str(n)] = group.shift(n).sort_index().values
        for w in windows:
            X['vma' + str(w)] = ma(group, w)
            X['vstd' + str(w)] = std(group, w)
        if amount is not None:
            for w in windows:
                X['vwap' + str(w)] = vwap(df[amount], df[volume], groupby=groupby, n=w)

    if (last_close is not None) and (price is not None):
        group = df[last_close].groupby([groupby])
        for n in range(raw_data):
            X[last_close + str(n)] = group.shift(n).sort_index().values
        for w in windows:
            # X['risk' + str(w)] = risk(df[price], df[last_close], groupby=groupby, n=w)
            X['kmid' + str(w)] = kmid(df[price], df[last_close], groupby=groupby, n=w)
        if (high is not None) and (low is not None):
            for w in windows:
                X['ksft' + str(w)] = ksft(df[price], df[last_close], df[high], df[low], groupby=groupby, n=w)

    if (high is not None) and (low is not None):
        group_h = df[high].groupby([groupby])
        group_l = df[low].groupby([groupby])
        for n in range(raw_data):
            X[high + str(n)] = group_h.shift(n).sort_index().values
            X[low + str(n)] = group_l.shift(n).sort_index().values
        for w in windows:
            X['hml' + str(w)] = hml(df[high], df[low], groupby=groupby, n=w)
        if price is not None:
            for w in windows:
                X['rsv' + str(w)] = rsv(df[price], df[high], df[low], groupby=groupby, n=w)
        if last_close is not None:
            for w in windows:
                X['klen' + str(w)] = klen(df[high], df[low], df[last_close], groupby=groupby, n=w)
    return X


def make_factors_series(kwargs, windows=None):
    """
    只比make_factors少了个groupby参数

    :param kwargs:
    {
        data: pd.DataFrame, 输入的数据
        price: str, 最新价格
        open: str, 最近的收盘价（昨天或者其它）
        volume: str, 当前tick的交易量
        amount: str, 当前tick的交易额
        high: str, 当前tick的最高价
        low: str, 当前tick的最低价
        label : str, 目标值
        shift: int, 应滞后的阶数，由label的构建方法决定
    }
    :param windows: list, 移动窗口的列表
    :return: pd.DataFrame
    """
    df = kwargs['data']
    price = kwargs['price']
    last_close = kwargs['open']
    volume = kwargs['volume']
    amount = kwargs['amount']
    high = kwargs['high']
    low = kwargs['low']
    label = kwargs['label']
    shift = kwargs['shift']
    if windows is None:
        windows = [5, 10, 20, 30, 60]

    X = pd.DataFrame()
    if label is not None:
        for w in windows:
            X['ma' + str(w)] = ma(df[label].shift(shift), w)
            X['std' + str(w)] = std(df[label].shift(shift), w)
            X['max' + str(w)] = max_(df[label].shift(shift), w)
            X['min' + str(w)] = min_(df[label].shift(shift), w)
            X['beta' + str(w)] = beta(df[label].shift(shift), w)
            X['roc' + str(w)] = roc(df[label].shift(shift), w)

    if price is not None:
        for w in windows:
            X['MA' + str(w)] = ma(df[price], w)
            X['STD' + str(w)] = std(df[price], w)
            X['MAX' + str(w)] = max_(df[price], w)
            X['MIN' + str(w)] = min_(df[price], w)
            X['BETA' + str(w)] = beta(df[price], w)
            X['ROC' + str(w)] = roc(df[price], w)

    if volume is not None:
        for w in windows:
            X['vma' + str(w)] = ma(df[volume], w)
            X['vstd' + str(w)] = std(df[volume], w)

    if (volume is not None) and (amount is not None):
        for w in windows:
            X['vwap' + str(w)] = vwap_series(df[amount], df[volume], n=w)

    if (last_close is not None) and (price is not None):
        for w in windows:
            X['risk' + str(w)] = risk_series(df[price], df[last_close], n=w)

    if (high is not None) and (low is not None):
        for w in windows:
            X['hml' + str(w)] = hml_series(df[high], df[low], n=w)

    if (price is not None) and (high is not None) and (low is not None):
        for w in windows:
            X['rsv' + str(w)] = rsv_series(df[price], df[high], df[low], n=w)
    return X
