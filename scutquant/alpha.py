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
import time


def cal_dif(prices, n=12, m=26):
    ema_n = prices.ewm(span=n, min_periods=n - 1).mean()
    ema_m = prices.ewm(span=m, min_periods=m - 1).mean()
    dif = ema_n - ema_m
    return dif


def cal_dea(dif, k=9):
    dea = dif.ewm(span=k, min_periods=k - 1).mean()
    return dea


def cal_rsi(price: pd.Series, windows: int = 14) -> pd.Series:
    # 计算每日涨跌情况
    diff = price.diff()
    up = diff.copy()
    up[up < 0] = 0
    down = diff.copy()
    down[down > 0] = 0

    # 计算RSI指标
    avg_gain = up.rolling(windows).sum() / windows
    avg_loss = down.abs().rolling(windows).sum() / windows
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def cal_psy(price: pd.Series, windows: int = 10) -> pd.Series:
    # 计算每日涨跌情况
    diff = price.diff()
    diff[diff > 0] = 1
    diff[diff <= 0] = 0

    # 计算PSY指标(其实是个分类指标, 在windows=w时, 最多有w+1个unique value(例如在windows=5时, 有0, 20, 40, 60, 80, 100))
    psy = diff.rolling(windows).sum() / windows * 100
    return psy


# 各大类特征
def MACD(X, data_group, groupby, name=None):
    # MACD中的DIF和DEA, 由于MACD是它们的线性组合所以没必要当作因子
    if name is None:
        name = ["DIF", "DEA"]
    features = pd.DataFrame()
    features[name[0]] = data_group.transform(lambda x: cal_dif(x))
    features[name[1]] = features[name[0]].groupby(groupby).transform(lambda x: cal_dea(x))
    return pd.concat([X, features], axis=1)


def RET(X, data, data_group, groupby, name="RET"):
    features = pd.DataFrame()
    for i in range(1, 5):
        features[name + "1_" + str(i)] = (data / data_group.shift(i) - 1)
        features[name + "2_" + str(i)] = (data / data_group.shift(i) - 1).groupby(groupby).rank(pct=True)
    return pd.concat([X, features], axis=1)


def SHIFT(X, data, data_group, windows, name):
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.shift(w) / data
    return pd.concat([X, features], axis=1)


def ROC(X, data, data_group, windows, name="ROC"):
    # https://www.investopedia.com/terms/r/rateofchange.asp
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = (data / data_group.shift(w) - 1) / w
    return pd.concat([X, features], axis=1)


def BETA(X, data, data_group, windows, name="BETA"):
    # The rate of close price change in the past d days, divided by latest close price to remove unit
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = (data - data_group.shift(w)) / (data * w)
    return pd.concat([X, features], axis=1)


def MA(X, data, data_group, windows, name="MA"):
    # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.transform(lambda x: x.rolling(w).mean()) / data
    return pd.concat([X, features], axis=1)


def STD(X, data, data_group, windows, name="STD"):
    # The standard deviation of close price for the past d days, divided by latest close price to remove unit
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.transform(lambda x: x.rolling(w).std()) / data
    return pd.concat([X, features], axis=1)


def MAX(X, data, data_group, windows, name="MAX"):
    # The max price for past d days, divided by latest close price to remove unit
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.transform(lambda x: x.rolling(w).max()) / data
    return pd.concat([X, features], axis=1)


def MIN(X, data, data_group, windows, name="MIN"):
    # The low price for past d days, divided by latest close price to remove unit
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.transform(lambda x: x.rolling(w).min()) / data
    return pd.concat([X, features], axis=1)


def QTL(X, data, data_group, windows, name="QTL"):
    # The x% quantile of past d day's close price, divided by latest close price to remove unit
    features = pd.DataFrame()
    for w in windows:
        features[name + "U" + str(w)] = data_group.transform(lambda x: x.rolling(w).quantile(0.8)) / data
        features[name + "D" + str(w)] = data_group.transform(lambda x: x.rolling(w).quantile(0.2)) / data
    return pd.concat([X, features], axis=1)


def CORR(X, data1_group, data2, windows, name="CORR"):
    # 受统计套利理论(股票配对交易)的启发，追踪个股收益率与大盘收益率的相关系数 这里的思路是: 如果近期(rolling=5, 10)的相关系数偏离了远期相关系数
    # (rolling=30, 60), 则有可能是个股发生了异动, 可根据异动的方向选择个股与大盘的多空组合
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data1_group.transform(lambda x: x.rolling(w).corr(data2))
    return pd.concat([X, features], axis=1)


def RSI(X, data_group, windows, name="RSI"):
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.transform(lambda x: cal_rsi(x, w))
    return pd.concat([X, features], axis=1)


def PSY(X, data_group, windows, name="PSY"):
    features = pd.DataFrame()
    for w in windows:
        features[name + str(w)] = data_group.transform(lambda x: cal_psy(x, w))
    return pd.concat([X, features], axis=1)


def PERF(X, data, group_idx, name="PERF"):
    # performance: 股票当日收益率相对大盘的表现
    features = pd.DataFrame()
    features[name + "1"] = data / group_idx.mean()
    features[name + "2"] = data / group_idx.std()
    features[name + "3"] = data / group_idx.max()
    features[name + "4"] = data / group_idx.min()
    return pd.concat([X, features], axis=1)


def IDX(X, data, idx, windows, name="IDX"):
    # 收盘价相对开盘价的变化, 与大盘的移动平均线对比
    features = pd.DataFrame()
    for w in windows:
        features[name + "1_" + str(w)] = data / idx.rolling(w).mean()
        features[name + "2_" + str(w)] = data / idx.rolling(w).max()
        features[name + "3_" + str(w)] = data / idx.rolling(w).min()
        features[name + "4_" + str(w)] = data / idx.rolling(w).median()
    return pd.concat([X, features], axis=1)


def RSV(X, data, low_group, high_group, windows, name="RSV"):
    # Represent the price position between upper and lower resistent price for past d days.
    features = pd.DataFrame()
    for w in windows:
        LOW = low_group.transform(lambda x: x.rolling(w).min())
        HIGH = high_group.transform(lambda x: x.rolling(w).max())
        features[name + str(w)] = (data - LOW) / (HIGH - LOW + 1e-12)
    return pd.concat([X, features], axis=1)


def DELTA(X, ret_group, idx_return, windows, name="DELTA"):
    # The delta of greek value
    features = pd.DataFrame()
    for w in windows:
        cov = ret_group.transform(lambda x: x.rolling(w).cov(idx_return))
        var = ret_group.transform(lambda x: x.rolling(w).var())
        features[name + str(w)] = cov / var
    return pd.concat([X, features], axis=1)

# todo: 增加其它希腊值: gamma, vega, rho, ...


def make_factors(kwargs=None, windows=None, fillna=False):
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
        }

        X = alpha.make_factors(kwargs)

    :param kwargs:
    {
        data: pd.DataFrame, 输入的数据
        close: str, 收盘价的列名
        open: str, 开盘价的列名
        volume: str, 当前tick的交易量
        amount: str, 当前tick的交易额
        high: str, 当前tick的最高价
        low: str, 当前tick的最低价
    }
    :param windows: list, 移动窗口的列表
    :param fillna: 是否填充缺失值
    :return: pd.DataFrame
    """
    start = time.time()
    if kwargs is None:
        kwargs = {}
    if "data" not in kwargs.keys():
        kwargs["data"] = pd.DataFrame()
    if "close" not in kwargs.keys():
        kwargs["close"] = None
    if "open" not in kwargs.keys():
        kwargs["open"] = None
    if "volume" not in kwargs.keys():
        kwargs["volume"] = None
    if "amount" not in kwargs.keys():
        kwargs["amount"] = None
    if "high" not in kwargs.keys():
        kwargs["high"] = None
    if "low" not in kwargs.keys():
        kwargs["low"] = None

    data = kwargs["data"]
    open = kwargs["open"]
    close = kwargs["close"]
    volume = kwargs['volume']
    amount = kwargs['amount']
    high = kwargs['high']
    low = kwargs['low']
    datetime = data.index.names[0]
    groupby = data.index.names[1]

    if windows is None:
        windows = [5, 10, 20, 30, 60]

    X = pd.DataFrame(index=data.index)

    # 先计算好分组再反复调用，节省重复计算花费的时间(实验表明可以节省约80%的时间)
    group_c = data[close].groupby(groupby) if close is not None else None
    group_o = data[open].groupby(groupby) if open is not None else None
    group_h = data[high].groupby(groupby) if high is not None else None
    group_l = data[low].groupby(groupby) if low is not None else None
    group_v = data[volume].groupby(groupby) if volume is not None else None
    group_a = data[amount].groupby(groupby) if amount is not None else None

    if close is not None:
        data["ret"] = data[close] / group_c.shift(1) - 1
        # data["ret"].fillna(1e-12)
        group_r = data["ret"].groupby(groupby)
        mean_ret = data["ret"].groupby(datetime).mean()

        X = MACD(X, group_c, groupby=groupby)
        X = RET(X, data[close], group_c, groupby=datetime)
        X = SHIFT(X, data[close], group_c, windows=windows, name="CLOSE")
        X = ROC(X, data[close], group_c, windows=windows)
        X = BETA(X, data[close], group_c, windows=windows)
        X = MA(X, data[close], group_c, windows=windows)
        X = STD(X, data[close], group_c, windows=windows)
        X = MAX(X, data[close], group_c, windows=windows)
        X = MIN(X, data[close], group_c, windows=windows)
        X = QTL(X, data[close], group_c, windows=windows)
        # X = MA(X, data["ret"], group_r, windows=windows, name="MA2_")
        # X = STD(X, data["ret"], group_r, windows=windows, name="STD2_")
        X = CORR(X, group_r, mean_ret, windows=windows)

        group_r_rank = X["RET2_1"].groupby(groupby)
        X = CORR(X, group_r_rank, mean_ret, windows=windows, name="CORR2_")

        # 来自行为金融学的指标
        X = RSI(X, group_c, windows=windows)
        X = PSY(X, group_c, windows=windows)

        # 来自金融工程的指标
        # X = DELTA(X, group_r, mean_ret, windows=windows)

        # X["PSY"] = data[close].groupby(groupby).transform(lambda x: cal_psy(x.rolling(14)))
        del mean_ret, group_r, group_r_rank

        if open is not None:
            chg_rate = data[close] / data[open] - 1
            group_idx = chg_rate.groupby(datetime)
            idx = group_idx.mean()

            X = PERF(X, chg_rate, group_idx)
            X = IDX(X, chg_rate, idx, windows=windows)

            features = pd.DataFrame()
            # 收盘价对开盘价的变化, 对于大盘的表现
            features["R_DELTA"] = (data[close] - data[open]).groupby(datetime).rank(pct=True)
            features["KMID"] = chg_rate

            del chg_rate, group_idx, idx

            if high is not None:
                features["KUP"] = (data[high] - data[open]) / data[open]
                if low is not None:
                    l9 = group_l.transform(lambda x: x.rolling(9).min())
                    h9 = group_h.transform(lambda x: x.rolling(9).max())
                    # KDJ指标
                    features["KDJ_K"] = (data[close] - l9) / (h9 - l9) * 100
                    features["KDJ_D"] = features["KDJ_K"].groupby(groupby).transform(lambda x: x.rolling(3).mean())
                    del l9, h9
                    features["KLEN"] = (data[high] - data[low]) / data[open]
                    features["KIMD2"] = (data[close] - data[open]) / (data[high] - data[low] + 1e-12)
                    features["KUP2"] = (data[high] - data[open]) / (data[high] - data[low] + 1e-12)
                    features["KLOW"] = (data[close] - data[low]) / data[open]
                    features["KLOW2"] = (data[close] - data[low]) / (data[high] - data[low] + 1e-12)
                    features["KSFT"] = (2 * data[close] - data[high] - data[low]) / data[open]
                    features["KSFT2"] = (2 * data[close] - data[high] - data[low]) / (data[high] - data[low] + 1e-12)
                    features["VWAP"] = (data[high] + data[low] + data[close]) / (3 * data[open])
                    X = pd.concat([X, features], axis=1)
                    X = RSV(X, data[close], group_l, group_h, windows=windows)
            del features
    if open is not None:
        X = SHIFT(X, data[open], group_o, windows=windows, name="OPEN")

    if high is not None:
        X = SHIFT(X, data[high], group_h, windows=windows, name="HIGH")
        if low is not None:
            if close is not None:
                X["MEAN1"] = (data[high] + data[low]) / (2 * data[close])

    if low is not None:
        X = SHIFT(X, data[low], group_l, windows=windows, name="LOW")

    if volume is not None:
        # data["chg_vol"] = data[volume] / group_v.shift(1) - 1
        # group_rv = data["chg_vol"].groupby(groupby)
        X = SHIFT(X, data[volume], group_v, windows=windows, name="VOLUME")
        X = MA(X, data[volume], group_v, windows=windows, name="VMA")
        X = STD(X, data[volume], group_v, windows=windows, name="VSTD")
        # X = MA(X, data["chg_vol"], group_rv, windows=windows, name="VMA2_")
        # X = STD(X, data["chg_vol"], group_rv, windows=windows, name="VSTD2_")
        X["VMEAN"] = data[volume] / data[volume].groupby(datetime).mean()
        if amount is not None:
            mean = data[amount] / data[volume]
            group_mean = mean.groupby(groupby)
            X["MEAN2"] = mean / mean.groupby(datetime).mean()
            X = SHIFT(X, mean, group_mean, windows=windows, name="MEAN2_")
            del mean, group_mean
        """
        if close is not None:
            group_r = data[close] / group_c.shift(1) - 1
            # 收益率和chg_vol的相关系数
            X = CORR(X, group_r, data["chg_vol"], windows=windows, name="CORRCV")
            del group_r
        """

    if amount is not None:
        X = SHIFT(X, data[amount], group_a, windows=windows, name="AMOUNT")

    if fillna:
        X = X.groupby(groupby).fillna(method="ffill").fillna(X.mean())
    end = time.time()
    print("time used:", end - start)
    return X


def alpha360(kwargs, shift=60, fillna=False):
    start = time.time()
    if kwargs is None:
        kwargs = {}
    if "data" not in kwargs.keys():
        kwargs["data"] = pd.DataFrame()
    if "close" not in kwargs.keys():
        kwargs["close"] = "close"
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

    data = kwargs["data"]
    open = kwargs["open"]
    close = kwargs["close"]
    volume = kwargs['volume']
    amount = kwargs['amount']
    high = kwargs['high']
    low = kwargs['low']
    groupby = data.index.names[1]

    X = pd.DataFrame()

    group_c = data[close].groupby(groupby) if close is not None else None
    group_o = data[open].groupby(groupby) if open is not None else None
    group_h = data[high].groupby(groupby) if high is not None else None
    group_l = data[low].groupby(groupby) if low is not None else None
    group_v = data[volume].groupby(groupby) if volume is not None else None
    group_a = data[amount].groupby(groupby) if amount is not None else None

    windows = [i for i in range(0, shift)]

    if open is not None:
        X = SHIFT(X, data[open], group_o, windows=windows, name="OPEN")

    if close is not None:
        X = SHIFT(X, data[close], group_c, windows=windows, name="CLOSE")

    if high is not None:
        X = SHIFT(X, data[high], group_h, windows=windows, name="HIGH")

    if low is not None:
        X = SHIFT(X, data[low], group_l, windows=windows, name="LOW")

    if volume is not None:
        X = SHIFT(X, data[volume], group_v, windows=windows, name="VOLUME")

    if amount is not None:
        X = SHIFT(X, data[amount], group_a, windows=windows, name="AMOUNT")

    if fillna:
        X = X.groupby(groupby).fillna(method="ffill").fillna(X.mean())
    end = time.time()
    print("time used:", end - start)
    return X
