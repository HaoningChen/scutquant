import pandas as pd
import numpy as np
from joblib import Parallel, delayed

"""

与qlib的将数据简单加工后扔给ai模型找规律的思路不同, worldquant的思路是用精细的operators挖到有逻辑, 且回测表现良好的因子, 
用因子值构造中性组合, 直接用于投资
即qlib的量化投资流程是: 数据 -> 因子(这里的因子更接近feature的概念) -> 模型 -> 策略 -> 收益
而worldquant的流程是: 数据 -> 因子 -> 收益, 策略就是根据因子值构建投资组合(参考scutquant.alpha.market_neutralize的注释)
其实可以将qlib中的模型预测值当作worldquant中的因子, 那么qlib其实用是一种单因子策略进行投资, 只不过因子是由ai模型挖掘的, 而且策略更加多样
而worldquant的每一个因子都代表某个策略, 一个portfolio manager会选择多个因子, 并分配不同资金给每一个因子, 最后所有因子收益加总得到portfolio
一言以蔽之, worldquant 模式是量化1.0时代的经典模式, 而qlib的模式则适用于量化2.0甚至3.0时代. 
但这并不意味着两者是不兼容的. 事实上, 一个被精细加工过的feature能让模型的预测效果更好, 反过来模型的预测值也能作为一个很好的因子素材

scutquant的alpha模块用的是qlib的思路, 而为了让用户按照worldquant的方式构造自己的因子, 本模块应运而生
在本模块中, ts是对每个instrument在时序上计算, 而cs是在截面上计算, 所有返回结果都是pd.Series
该模块提供了更加丰富的算子, 且速度也在不断优化. 计划以后alpha只提供因子表达式, 而具体计算由operators的算子完成
未来这部分可能会合并到alpha模块中, 让整个架构看起来不那么臃肿, 但也要考虑到合并后是否方便维护的问题

example:  

from operators import * 

factor = cs_zscore(ts_rank(ts_corr(df, "close", "volume", 15), 15))

"""


def ts_delay(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns data n_period days ago
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.shift(n_period))
    else:
        res: pd.Series = data.transform(lambda x: x.shift(n_period))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_delta(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns data - ts_delay(data, n_period)
    """
    return data - ts_delay(data, n_period)


def ts_returns(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns the relative change of data .
    :return:
    """
    return ts_delta(data, n_period) / ts_delay(data, n_period)


def ts_sum(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Sum values of data for the past n_period days.
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).sum())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_product(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns product of data for the past n_period days
    """
    if isinstance(data, pd.Series):
        prod = data.groupby(level=1).transform(lambda x: x.cumprod())
    else:
        prod = data.transform(lambda x: x.cumprod())
        prod.index.names = ["datetime", "instrument"]
    return prod / ts_delay(data, n_period)


def ts_max(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns max value of data for the past n_period days
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).max())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).max())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_min(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns min value of data for the past n_period days
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).min())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).min())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_mean(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns average value of data for the past n_period days.
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).mean())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).mean())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_ewma(data: pd.core.groupby.SeriesGroupBy | pd.Series, a: float) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.ewm(alpha=a).mean())
    else:
        res: pd.Series = data.transform(lambda x: x.ewm(alpha=a).mean())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_std(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns standard deviation of data for the past n_period days
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).std())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).std())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_dstd(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns downside standard deviation of data for the past n_period days
    """

    def downside_std(df: pd.Series):
        downside_data = df.where(df > 0, np.nan)
        return downside_data.rolling(n_period, min_periods=2).std()

    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: downside_std(x))
    else:
        res: pd.Series = data.transform(lambda x: downside_std(x))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_kurt(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns kurtosis of data for the last n_period days.
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).kurt())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).kurt())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_skew(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Return skewness of data for the past n_period days.
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).skew())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).skew())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_median(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns median value of data for the past n_period days
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).median())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).median())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_rank(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Rank the values of data for each instrument over the past n_period days, then return the rank of the current value
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).rank(pct=True))
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).rank(pct=True))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_variance(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Returns variance of data for the past n_period days
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).var())
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).var())
        res.index.names = ["datetime", "instrument"]
        return res


def ts_quantile_up(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).quantile(0.75))
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).quantile(0.75))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_quantile_down(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: x.rolling(n_period).quantile(0.25))
    else:
        res: pd.Series = data.transform(lambda x: x.rolling(n_period).quantile(0.25))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_zscore(data: pd.Series, n_period: int) -> pd.Series:
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean:
    (data - ts_mean(data,n_period)) / ts_std(data,n_period)
    """
    return (data - ts_mean(data, n_period)) / ts_std(data, n_period)


def ts_robust_zscore(data: pd.Series, n_period: int) -> pd.Series:
    med = ts_median(data, n_period)
    return (data - med) / (abs(med) * 1.4826)


def ts_scale(data: pd.Series, n_period: int) -> pd.Series:
    return (data - ts_min(data, n_period)) / (ts_max(data, n_period) - ts_min(data, n_period))


def ts_sharpe(data: pd.core.groupby.SeriesGroupBy | pd.Series, n_period: int) -> pd.Series:
    """
    Return sharpe ratio ts_mean(data, n_period) / ts_std(data, n_period)
    """
    return ts_mean(data, n_period) / ts_std(data, n_period)


def ts_av_diff(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns data - ts_mean(data, n_period)
    """
    return data - ts_mean(data, n_period)


def ts_max_diff(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns data - ts_max(data, n_period)
    """
    return data - ts_max(data, n_period)


def ts_min_diff(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns data - ts_min(data, n_period)
    """
    return data - ts_min(data, n_period)


def ts_corr(data: pd.DataFrame, feature: str, label: str, n_period: int) -> pd.Series:
    """
    Returns correlation of data[feature] and data[label] for the past n_period days

    :param data: pd.DataFrame, must include columns feature and label
    :param feature:
    :param label:
    :param n_period:
    :return:
    """
    corr = data.groupby(level=1).apply(lambda x: x[feature].rolling(n_period).corr(x[label])).reset_index(0, drop=True)
    return corr.sort_index()


def ts_cov(data: pd.DataFrame, feature: str, label: str, n_period: int) -> pd.Series:
    """
    Returns covariance of data[feature] and data[label] for the past n_period days

    :param data: pd.DataFrame, must include columns feature and label
    :param feature:
    :param label:
    :param n_period:
    :return:
    """
    cov = data.groupby(level=1).apply(lambda x: x[feature].rolling(n_period).cov(x[label])).reset_index(0, drop=True)
    return cov.sort_index()


def ts_beta(data: pd.DataFrame, feature: str, label: str, n_period: int) -> pd.Series:
    """
    Returns beta of data[feature] and data[label] for the past n_period days

    :param data: pd.DataFrame, must include columns feature and label
    :param feature:
    :param label:
    :param n_period:
    :return:
    """
    cov = ts_cov(data, feature, label, n_period)
    var = ts_variance(data[feature], n_period)
    return cov / var


def ts_regression(data: pd.DataFrame, feature: str, label: str, n_periods: int, rettype: int = 0) -> pd.Series:
    """
    Returns results of linear model y = beta * x + alpha + resid

    :param data:
    :param feature:
    :param label:
    :param n_periods:
    :param rettype: 0 for resid, 1 for beta, 2 for alpha, 3 for y_hat, 4 for R^2
    :return:
    """
    if rettype == 0:
        beta = ts_beta(data, feature, label, n_periods)
        alpha: pd.Series = ts_mean(data[label], n_periods) - beta * ts_mean(data[feature], n_periods)
        predict: pd.Series = beta * data[feature] + alpha
        resid: pd.Series = data[label] - predict
        return resid
    elif rettype == 1:
        beta = ts_beta(data, feature, label, n_periods)
        return beta
    elif rettype == 2:
        beta = ts_beta(data, feature, label, n_periods)
        alpha: pd.Series = ts_mean(data[label], n_periods) - beta * ts_mean(data[feature], n_periods)
        return alpha
    elif rettype == 3:
        beta = ts_beta(data, feature, label, n_periods)
        alpha: pd.Series = ts_mean(data[label], n_periods) - beta * ts_mean(data[feature], n_periods)
        predict: pd.Series = beta * data[feature] + alpha
        return predict
    else:
        beta = ts_beta(data, feature, label, n_periods)
        alpha: pd.Series = ts_mean(data[label], n_periods) - beta * ts_mean(data[feature], n_periods)
        predict: pd.Series = beta * data[feature] + alpha
        predict.name = "predict"
        concat_df = pd.concat([predict, data[label]], axis=1)
        return ts_corr(concat_df, "predict", label, n_periods) ** 2


def ts_pos_count(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns the number of days when data is bigger than 0 for the past n_period days

    psy = ts_pos_count(ts_delta(close, 1), d) / d * 100  # 一个计算d日psy指标的例子, 比alpha模块的对应函数简洁了不少
    """
    data_copy = data.copy()
    data_copy[data_copy > 0] = 1
    data_copy[data_copy < 0] = 0
    return data_copy.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())


def ts_neg_count(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns the number of days when data is smaller than 0 for the past n_period days
    """
    data_copy = data.copy()
    data_copy[data_copy > 0] = 0
    data_copy[data_copy < 0] = 1
    return data_copy.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())


def linear_decay(x: pd.Series, window: int) -> pd.Series:
    """
    Applies linear decay to a time series.

    :param x: The time series to apply linear decay to.
    :type x: pd.Series
    :param window: The window size for the linear decay.
    :type window: int
    :return: The time series with linear decay applied.
    :rtype: pd.Series
    """
    weights = [np.exp(-1 / window * (window - t)) for t in range(window)]
    return x.rolling(window).apply(lambda y: sum(y * weights) / sum(weights), raw=True)


def ts_decay_linear(data: pd.Series | pd.core.groupby.SeriesGroupBy, n_period: int) -> pd.Series:
    """
    Returns the linear decay on data for the past n_period days.
    """
    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: linear_decay(x, n_period))
    else:
        res: pd.Series = data.transform(lambda x: linear_decay(x, n_period))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_argmax(data: pd.Series | pd.core.groupby.SeriesGroupBy, n_period: int) -> pd.Series:
    def argmax(feature: pd.Series) -> pd.Series:
        return feature.rolling(n_period).apply(lambda x: np.argmax(x))

    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: argmax(x))
    else:
        res = data.transform(lambda x: argmax(x))
        res.index.names = ["datetime", "instrument"]
        return res


def ts_argmin(data: pd.Series | pd.core.groupby.SeriesGroupBy, n_period: int) -> pd.Series:
    def argmin(feature: pd.Series) -> pd.Series:
        return feature.rolling(n_period).apply(lambda x: np.argmin(x))

    if isinstance(data, pd.Series):
        return data.groupby(level=1).transform(lambda x: argmin(x))
    else:
        res = data.transform(lambda x: argmin(x))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_rank(data: pd.core.groupby.SeriesGroupBy | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0.
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: x.rank(pct=True))
    else:
        res: pd.Series = data.transform(lambda x: x.rank(pct=True))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_zscore(data: pd.core.groupby.SeriesGroupBy | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
    else:
        res: pd.Series = data.transform(lambda x: (x - x.mean()) / x.std())
        res.index.names = ["datetime", "instrument"]
        return res


def cs_robust_zscore(data: pd.core.groupby.SeriesGroupBy | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: (x - x.median()) / (abs(x.median()) * 1.4826))
    else:
        res: pd.Series = data.transform(lambda x: (x - x.median()) / (abs(x.median()) * 1.4826))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_scale(data: pd.core.groupby.SeriesGroupBy | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        res: pd.Series = data.transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        res.index.names = ["datetime", "instrument"]
        return res


def cs_mean(data: pd.core.groupby.SeriesGroupBy | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    This function is not for regular alphas which have two index levels. It calculates the mean value of all instruments
    on a particular time tick. You may use this for calculating the relationship between single instrument and the index
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return data.groupby(level=0).transform(lambda x: x.mean())
    else:
        res: pd.Series = data.transform(lambda x: x.mean())
        res.index.names = ["datetime", "instrument"]
        return res


def cs_shrink(data: pd.core.groupby.SeriesGroupBy | pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.groupby(level=0).transform(lambda x: x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5))
        data = data.groupby(level=0).transform(lambda x: x.where(x >= -3, -3 + (x + 3).div(x.min() + 3) * 0.5))
        return data
    else:
        res = data.transform(lambda x: x.where(x <= 3, 3 + (x - 3).div(x.max() - 3) * 0.5))
        res = res.transform(lambda x: x.where(x >= -3, -3 + (x + 3).div(x.min() + 3) * 0.5))
        res.index.names = ["datetime", "instrument"]
        return res


def sign(data: pd.Series) -> pd.Series:
    return pd.Series(np.sign(data.values), index=data.index)


def sign_power(data: pd.Series, p: float) -> pd.Series:
    return sign(data) * (abs(data) ** p)


def log(data: pd.Series) -> pd.Series:
    return pd.Series(np.log(data.values), index=data.index)


def tanh(data: pd.Series) -> pd.Series:
    return pd.Series(np.tanh(data.values), index=data.index)


def sigmoid(data: pd.Series) -> pd.Series:
    return pd.Series(1 / (1 + np.exp(-data.values)), index=data.index)


def bigger(data1: pd.Series, data2: pd.Series) -> pd.Series:
    """
    Returns the bigger value of data1 and data2
    """
    return data1.where(data1 < data2, data1)  # 若不满足data1 < data2, 则返回data1


def smaller(data1: pd.Series, data2: pd.Series) -> pd.Series:
    """
    Returns the smaller value of data1 and data2
    """
    return data1.where(data1 > data2, data1)


def inf_mask(data: pd.Series) -> pd.Series:
    """
    Replace inf with nan
    """
    return data.where(data != np.inf, np.nan)


def get_resid(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    经过百万级的数据的上千次实验, 发现此方法比调用sklearn.linear_model的LinearRegression平均快一倍
    """
    cov = x.cov(y)
    var = x.var()
    beta = cov / var
    del cov, var
    beta0 = y.mean() - beta * x.mean()
    return y - beta0 - beta * x


def neutralize(data: pd.DataFrame | pd.Series, target: pd.Series, features: list[str] = None,
               n_jobs=-1) -> pd.DataFrame | pd.Series:
    """
    在截面上对选定的features进行target中性化, 剩余因子不变

    example:

    # 使用补充数据data, 对factor_raw的RSI, MACD和KDJ_K因子进行市值中性化

    factor_neutralized = alpha.neutralize(factor_raw, target=data["ln_market_value"], features=["RSI", "MACD", "KDJ_K"])

    :param data: 需要中性化的因子集合
    :param target: 解释变量
    :param features: 需要中性化的因子名(列表), 因为不同因子可能需要不同的中性化手法, 故通过此参数控制进行中性化的因子
    :param n_jobs: 同时调用的cpu数
    :return: pd.DataFrame, 包括中性化后的因子和未中性化的其它因子
    """
    RETTYPE = "df"

    def neutralize_single_factor(f_name: str) -> pd.Series:
        result = concat_data[[f_name, target_name]].groupby(level=0, group_keys=False).apply(
            lambda x: get_resid(x[target_name], x[f_name]))
        result.name = f_name
        return result

    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name)
        RETTYPE = "series"

    target = target[target.index.isin(data.index)]
    concat_data = pd.concat([data, target], axis=1)
    target_name = target.name
    features = data.columns if features is None else features
    other_cols = [c for c in data.columns if c not in features]
    del data, target

    factor_neu = Parallel(n_jobs=n_jobs)(delayed(neutralize_single_factor)(f) for f in features)
    data_neu = pd.concat(factor_neu, axis=1)
    del factor_neu

    if RETTYPE == "df":
        return pd.concat([data_neu, concat_data[other_cols]], axis=1)
    else:
        return data_neu.iloc[:, 0]
