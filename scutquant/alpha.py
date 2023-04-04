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
import numpy as np
import time


def ewm(x):
    # exp_weighted_mean
    a = 1 - 2 / (1 + len(x))
    w = a ** np.arange(len(x))[::-1]
    w /= w.sum()
    return np.nansum(w * x)


def ema(X, groupby, window):
    return X.groupby(groupby).transform(lambda x: x.rolling(window).apply(ewm))


def make_factors(kwargs=None, windows=None, use_macd_features=False):
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
    :param use_macd_features: 是否计算MACD, 计算会消耗大量时间
    :return: pd.DataFrame
    """
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
    datetime = data.index.names[0]
    groupby = data.index.names[1]

    if windows is None:
        windows = [5, 10, 20, 30, 60]

    X = pd.DataFrame(index=data.index)

    if close is not None:
        data["ret"] = data[close].groupby(groupby).shift(1) / data[close] - 1
        mean_ret = data["ret"].groupby(datetime).mean()
        for i in range(1, 5):
            X["RET1_" + str(i)] = (data[close].groupby(groupby).shift(i) / data[close] - 1)
            X["RET2_" + str(i)] = (data[close].groupby(groupby).shift(i) / data[close] - 1).groupby(datetime).rank(
                pct=True)
        for w in windows:
            X["CLOSE" + str(w)] = data[close].groupby(groupby).shift(w) / data[close]
            # https://www.investopedia.com/terms/r/rateofchange.asp
            X["ROC" + str(w)] = (data[close] - data[close].groupby(groupby).shift(w) - 1) / w
            # The rate of close price change in the past d days, divided by latest close price to remove unit
            X["BETA" + str(w)] = (data[close] - data[close].groupby(groupby).shift(w)) / (data[close] * w)
            # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
            X["MA" + str(w)] = data[close].groupby(groupby).transform(lambda x: x.rolling(w).mean()) / data[close]
            # The standard diviation of close price for the past d days, divided by latest close price to remove unit
            X["STD" + str(w)] = data[close].groupby(groupby).transform(lambda x: x.rolling(w).std()) / data[close]
            # The max price for past d days, divided by latest close price to remove unit
            X["MAX" + str(w)] = data[close].groupby(groupby).transform(lambda x: x.rolling(w).max()) / data[close]
            # The low price for past d days, divided by latest close price to remove unit
            X["MIN" + str(w)] = data[close].groupby(groupby).transform(lambda x: x.rolling(w).min()) / data[close]
            # The 80% quantile of past d day's close price, divided by latest close price to remove unit
            X["QTLU" + str(w)] = data[close].groupby(groupby).transform(lambda x: x.rolling(w).quantile(0.8)) / data[
                close]
            # The 20% quantile of past d day's close price, divided by latest close price to remove unit
            X["QTLD" + str(w)] = data[close].groupby(groupby).transform(lambda x: x.rolling(w).quantile(0.2)) / data[
                close]
            # 受统计套利理论(股票配对交易)的启发，追踪个股收益率与大盘收益率的相关系数
            # 这里的思路是: 如果近期(rolling=5, 10)的相关系数偏离了远期相关系数(rolling=30, 60), 则有可能是个股发生了异动,
            # 可根据异动的方向选择个股与大盘的多空组合
            X["CORR" + str(w)] = data["ret"].groupby(groupby).transform(lambda x: x.rolling(w).corr(mean_ret.rolling(w)))
        del data["ret"]
        del mean_ret
        if use_macd_features:
            # 一眼丁真, 鉴定为垃圾因子
            # 怀疑跟MA因子相关性太高导致失效
            print("Processing MACD factors...")
            X["DIF"] = (ema(data[close], groupby, 12) - ema(data[close], groupby, 26)) / data[close]
            print("DIF done")
            X["DEA"] = ema(X["DIF"], groupby, 9) / data[close]
            print("DEA done")
            X["MACD"] = 2 * (X["DIF"] - X["DEA"])
            print("MACD done")

        if open is not None:
            X["DELTA"] = (data[close] - data[open]).groupby(datetime).rank(pct=True)
            X["KMID"] = data[close] / data[open] - 1
            # performance: 股票当日收益率相对大盘的表现
            X["PERF1"] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(datetime).mean()
            X["PERF2"] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(datetime).max()
            X["PERF3"] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(datetime).min()
            X["PERF4"] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(datetime).median()
            for w in windows:
                # 股票收盘对开盘的收益, 与大盘移动平均线相比的强弱
                X["IDX1_" + str(w)] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(
                    datetime).mean().rolling(w).mean()
                X["IDX2_" + str(w)] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(
                    datetime).mean().rolling(w).max()
                X["IDX3_" + str(w)] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(
                    datetime).mean().rolling(w).min()
                X["IDX4_" + str(w)] = (data[close] / data[open] - 1) / (data[close] / data[open] - 1).groupby(
                    datetime).mean().rolling(w).median()
            if high is not None:
                X["KUP"] = (data[high] - data[open]) / data[open]
                if low is not None:
                    l9 = data[low].groupby(groupby).transform(lambda x: x.rolling(9).min())
                    h9 = data[high].groupby(groupby).transform(lambda x: x.rolling(9).max())
                    # KDJ指标
                    X["KDJ_K"] = (data[close] - l9) / (h9 - l9) * 100
                    X["KDJ_D"] = X["KDJ_K"].groupby(groupby).transform(lambda x: x.rolling(3).mean())
                    # X["KDJ_J"] = 3 * X["KDJ_D"] - 2 * X["KDJ_K"]  # K和D的线性组合，没必要加上
                    del l9
                    del h9
                    X["KLEN"] = (data[high] - data[low]) / data[open]
                    X["KIMD2"] = (data[close] - data[open]) / (data[high] - data[low] + 1e-12)
                    X["KUP2"] = (data[high] - data[open]) / (data[high] - data[low] + 1e-12)
                    X["KLOW"] = (data[close] - data[low]) / data[open]
                    X["KLOW2"] = (data[close] - data[low]) / (data[high] - data[low] + 1e-12)
                    X["KSFT"] = (2 * data[close] - data[high] - data[low]) / data[open]
                    X["KSFT2"] = (2 * data[close] - data[high] - data[low]) / (data[high] - data[low] + 1e-12)
                    X["VWAP"] = (data[high] + data[low] + data[close]) / (3 * data[open])
                    for w in windows:
                        LOW = data[low].groupby(groupby).transform(lambda x: x.rolling(w).min())
                        HIGH = data[high].groupby(groupby).transform(lambda x: x.rolling(w).max())
                        # Represent the price position between upper and lower resistent price for past d days.
                        X["RSV" + str(w)] = (data[close] - LOW) / (HIGH - LOW + 1e-12)
    if open is not None:
        for w in windows:
            X["OPEN" + str(w)] = data[open].groupby(groupby).shift(w) / data[open]
    if high is not None:
        for w in windows:
            X["HIGH" + str(w)] = data[high].groupby(groupby).shift(w) / data[high]
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            X["IMAX" + str(w)] = 100 * (w - data[high].groupby(groupby).transform(lambda x: x.rolling(w).max())) / (
                    data[close] * w)
        if low is not None:
            for w in windows:
                IMAX = 100 * (w - data[high].groupby(groupby).transform(lambda x: x.rolling(w).max())) / w
                IMIN = 100 * (w - data[low].groupby(groupby).transform(lambda x: x.rolling(w).min())) / w
                # The time period between previous lowest-price date occur after highest price date.
                X["IMXD" + str(w)] = (IMAX - IMIN) / data[close]
            if close is not None:
                X["MEAN1"] = (data[high] + data[low]) / (2 * data[close])
    if low is not None:
        for w in windows:
            X["LOW" + str(w)] = data[low].groupby(groupby).shift(w) / data[low]
            # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
            X["IMIN" + str(w)] = 100 * (w - data[low].groupby(groupby).transform(lambda x: x.rolling(w).min())) / (
                    data[close] * w)
    if volume is not None:
        for w in windows:
            X["VOLUME" + str(w)] = data[volume].groupby(groupby).shift(w) / data[volume]
            # https://www.barchart.com/education/technical-indicators/volume_moving_average
            X["VMA" + str(w)] = data[volume].groupby(groupby).transform(lambda x: x.rolling(w).mean()) / data[volume]
            # The standard deviation for volume in past d days.
            X["VSTD" + str(w)] = data[volume].groupby(groupby).transform(lambda x: x.rolling(w).std()) / data[volume]
        X["VMEAN"] = data[volume] / data[volume].groupby(datetime).mean()
        if amount is not None:
            mean = data[amount] / data[volume]
            X["MEAN2"] = mean / mean.groupby(datetime).mean()
            for w in windows:
                X["MEAN2_" + str(w)] = mean.groupby(groupby).shift(w) / mean
            del mean
    if amount is not None:
        for w in windows:
            X["AMOUNT" + str(w)] = data[amount].groupby(groupby).shift(w) / data[amount]
    end = time.time()
    print("time used:", end - start)
    return X


def alpha360(kwargs, shift=60):
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
    if open is not None:
        group = data[open].groupby(groupby)
        for i in range(1, shift + 1):
            X[open + str(i)] = group.shift(i) / data[close]

    if close is not None:
        group = data[close].groupby(groupby)
        for i in range(1, shift + 1):
            X[close + str(i)] = group.shift(i) / data[close]

    if high is not None:
        group = data[high].groupby(groupby)
        for i in range(1, shift + 1):
            X[high + str(i)] = group.shift(i) / data[close]

    if low is not None:
        group = data[low].groupby(groupby)
        for i in range(1, shift + 1):
            X[low + str(i)] = group.shift(i) / data[close]

    if volume is not None:
        group = data[volume].groupby(groupby)
        for i in range(1, shift + 1):
            X[volume + str(i)] = group.shift(i) / data[volume]

    if amount is not None:
        group = data[amount].groupby(groupby)
        for i in range(1, shift + 1):
            X[amount + str(i)] = group.shift(i) / (data[close] * data[volume])
    end = time.time()
    print("time used:", end - start)
    return X
