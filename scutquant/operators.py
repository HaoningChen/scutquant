import pandas as pd
import numpy as np

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
该模块提供了更加丰富的算子, 但批量计算因子时速度远远慢于alpha模块, 主要原因是该模块需要反复地groupby, 而alpha模块算一类因子只用groupby一次
未来这部分可能会合并到alpha模块中, 让整个架构看起来不那么臃肿, 但也要考虑到合并后是否方便维护的问题

example:  

from operators import * 

factor = cs_zscore(ts_rank(ts_corr(df, "close", "volume", 15), 15))

"""


def ts_delay(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns data n_period days ago
    """
    return data.groupby(level=1).transform(lambda x: x.shift(n_period))


def ts_delta(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns data - ts_delay(data, n_period)
    """
    return data - ts_delay(data, n_period)


def ts_sum(data: pd.Series, n_period: int) -> pd.Series:
    """
    Sum values of data for the past n_period days.
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).sum())


def ts_max(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns max value of data for the past n_period days
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).max())


def ts_min(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns min value of data for the past n_period days
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).min())


def ts_mean(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns average value of data for the past n_period days.
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).mean())


def ts_ewma(data: pd.Series, a: float = 0.94) -> pd.Series:
    return data.groupby(level=1).transform(lambda x: x.ewm(alpha=a).mean())


def ts_std(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns standard deviation of data for the past n_period days
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).std())


def ts_kurt(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns kurtosis of data for the last n_period days.
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).kurt())


def ts_skew(data: pd.Series, n_period: int) -> pd.Series:
    """
    Return skewness of data for the past n_period days.
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).skew())


def ts_median(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns median value of data for the past n_period days
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).median())


def ts_rank(data: pd.Series, n_period: int) -> pd.Series:
    """
    Rank the values of data for each instrument over the past n_period days, then return the rank of the current value
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).rank(pct=True))


def ts_variance(data: pd.Series, n_period: int) -> pd.Series:
    """
    Returns variance of data for the past n_period days
    """
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).var())


def ts_quantile_up(data: pd.Series, n_period: int) -> pd.Series:
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).quantile(0.75))


def ts_quantile_down(data: pd.Series, n_period: int) -> pd.Series:
    return data.groupby(level=1).transform(lambda x: x.rolling(n_period).quantile(0.25))


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


def ts_sharpe(data: pd.Series, n_period: int) -> pd.Series:
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
    beta = ts_beta(data, feature, label, n_periods)
    alpha: pd.Series = ts_mean(data[label], n_periods) - beta * ts_mean(data[feature], n_periods)
    predict: pd.Series = beta * data[feature] + alpha
    resid: pd.Series = data[label] - predict
    if rettype == 0:
        return resid
    elif rettype == 1:
        return beta
    elif rettype == 2:
        return alpha
    elif rettype == 3:
        return predict
    else:
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


def cs_rank(data: pd.Series) -> pd.Series:
    """
    Ranks the input among all the instruments and returns an equally distributed number between 0.0 and 1.0.
    """
    return data.groupby(level=0).transform(lambda x: x.rank(pct=True))


def cs_zscore(data: pd.Series) -> pd.Series:
    """
    Z-score is a numerical measurement that describes a value's relationship to the mean of a group of values.
    Z-score is measured in terms of standard deviations from the mean
    """
    return data.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())


def cs_robust_zscore(data: pd.Series) -> pd.Series:
    med = data.groupby(level=0).median()
    return (data - med) / (abs(med) * 1.4826)


def cs_scale(data: pd.Series) -> pd.Series:
    return data.groupby(level=0).transform(lambda x: (x - x.min()) / (x.max() - x.min()))


def cs_mean(data: pd.Series) -> pd.Series:
    """
    This function is not for regular alphas which have two index levels. It calculates the mean value of all instruments
    on a particular time tick. You may use this for calculating the relationship between single instrument and the index
    """
    return data.groupby(level=0).transform(lambda x: x.mean())


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
    return data1.where(data1 > data2, data2)


def smaller(data1: pd.Series, data2: pd.Series) -> pd.Series:
    """
    Returns the smaller value of data1 and data2
    """
    return data1.where(data1 < data2, data1)
