from .operators import *
from .report import calc_drawdown
import pandas as pd


def factor_neutralize(factors: pd.DataFrame | pd.Series, target: pd.Series,
                      feature: list[str] = None) -> pd.DataFrame | pd.Series:
    return neutralize(factors, features=feature, target=target)


def market_neutralize(x: pd.Series, long_only: bool = False) -> pd.Series:
    """
    市场组合中性化:
    (1) 对所有股票减去其截面上的因子均值
    (2) 在(1)之后, 对每支股票除以截面上的因子值绝对值之和

    这样处理后每支股票会获得一个权重, 代表着资金的方向和数量(例如0.5代表半仓做多, -0.25代表1/4仓做空),
    且截面上的权重之和为0, 绝对值之和为1.
    """
    _mean = x.groupby(level=0).mean()
    x -= _mean
    abs_sum = abs(x).groupby(level=0).sum()
    x /= abs_sum
    if long_only:
        x[x < 0] = 0
        x *= 2
    return x


def calc_factor_turnover(x: pd.Series) -> pd.Series:
    factor_neu = market_neutralize(x, long_only=False)
    instrument_to = abs(factor_neu - ts_delay(factor_neu, 1).fillna(0))
    return instrument_to.groupby(level=0).sum()


def get_factor_portfolio(feature: pd.Series, label: pd.Series, long_only: bool = False) -> pd.Series:
    """
    :param feature: 因子值
    :param label: 收益率
    :param long_only: 是否只做多
    :return: 时序数据portfolio, 代表累计收益率
    """
    x_neu = market_neutralize(feature, long_only=long_only)
    X = pd.DataFrame({"feature": x_neu, "label": label})
    X.dropna(inplace=True)
    X["factor_return"] = X["feature"] * X["label"]
    daily_return = X["factor_return"].groupby("datetime").sum()
    daily_return += 1
    portfolio = daily_return.cumprod()
    portfolio.index = pd.to_datetime(portfolio.index)
    return portfolio


def calc_fitness(sharpe: float, returns: float, turnover: float) -> float:
    """
    参考 https://platform.worldquantbrain.com/learn/documentation/discover-brain/intermediate-pack-part-1
    """
    return sharpe * ((abs(returns) / max(turnover, 0.125)) ** 0.5)


def get_factor_metrics(factor: pd.Series, label: pd.Series, metrics=None, handle_nan: bool = True,
                       long_only: bool = False) -> dict:
    """
    :param factor:
    :param label:
    :param metrics: list[str] = ["ic", "return", "turnover", "sharpe", "ir", "fitness"] 有些指标的计算必须依赖其它指标
    :param handle_nan:
    :param long_only: 是否只做多
    :return:
    """
    if metrics is None:
        metrics = ["ic", "return", "turnover", "sharpe", "ir", "fitness"]
    if handle_nan:
        label.dropna(inplace=True)
        factor = factor[factor.index.isin(label.index)]
        label = label[label.index.isin(factor.index)]
    result: dict = {}
    if "ic" in metrics:  # information coefficient
        result["ic"] = cs_corr(factor, label, rank=True).groupby(level=0).mean()
        result["ic_mean"] = result["ic"].mean()
        result["icir"] = result["ic"].mean() / result["ic"].std()
        result["t-stat"] = result["icir"] * (len(result["ic"]) ** 0.5)
    if "return" in metrics:
        result["return"] = get_factor_portfolio(factor, label, long_only=long_only)
        benchmark: pd.Series = label.groupby(level=0).mean()
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark = benchmark[benchmark.index.isin(result["return"].index)]
        result["benchmark"] = (benchmark + 1).cumprod()
        result["excess_return"] = result["return"] - result["benchmark"]
        result["drawdown"] = calc_drawdown(result["return"])
        result["excess_return_drawdown"] = calc_drawdown(result["excess_return"])
        result["max_drawdown"] = result["drawdown"].min()
        result["excess_return_max_drawdown"] = result["excess_return_drawdown"].min()
    if "turnover" in metrics:
        result["turnover"] = calc_factor_turnover(factor)
    if "sharpe" in metrics:
        result["sharpe"] = (result["return"].mean() - 1) / result["return"].std()
    if "ir" in metrics:  # information ratio
        result["ir"] = result["excess_return"].mean() / result["excess_return"].std()
    if "fitness" in metrics:
        result["fitness"] = calc_fitness(result["sharpe"], result["return"].mean(), result["turnover"].mean())
    result["return"] -= 1
    result["benchmark"] -= 1
    return result


class Alpha:
    def __init__(self):
        self.data = None
        self.norm_method = None
        self.process_nan = None
        self.result = None

    def call(self):
        """
        Write your alpha formula here
        """
        pass

    def normalize(self):
        self.result = mad_winsor(inf_mask(self.result))
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method=self.process_nan).fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class MA(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_mean(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["ma" + str(d)] = ts_mean(self.data, d)


class STD(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_std(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["std" + str(d)] = ts_std(self.data, d)


class KURT(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_kurt(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["kurt" + str(d)] = ts_kurt(self.data, d)


class SKEW(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_skew(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["skew" + str(d)] = ts_skew(self.data, d)


class DELAY(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_delay(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["delay" + str(d)] = ts_delay(self.data, d)


class DELTA(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_delta(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["delta" + str(d)] = ts_delta(self.data, d)


class MAX(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_max(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["max" + str(d)] = ts_max(self.data, d)


class MIN(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_min(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["min" + str(d)] = ts_min(self.data, d)


class RANK(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_rank(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["rank" + str(d)] = ts_rank(self.data, d)


class QTLU(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_quantile_up(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["qtlu" + str(d)] = ts_quantile_up(self.data, d)


class QTLD(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_quantile_down(self.data, self.periods)
        else:
            for d in self.periods:
                self.result["qtld" + str(d)] = ts_quantile_down(self.data, d)


class CORR(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_corr(self.feature, self.label, self.periods)
        else:
            for d in self.periods:
                self.result["corr" + str(d)] = ts_corr(self.feature, self.label, d)


class CORD(Alpha):
    # The correlation between feature change ratio and label change ratio
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        fd = self.feature / ts_delay(self.feature, 1)
        ld = self.label / ts_delay(self.label, 1)
        if isinstance(self.periods, int):
            self.result = ts_corr(fd, ld, self.periods)
        else:
            for d in self.periods:
                self.result["cord" + str(d)] = ts_corr(fd, ld, d)


class COV(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_cov(self.feature, self.label, self.periods)
        else:
            for d in self.periods:
                self.result["cov" + str(d)] = ts_cov(self.feature, self.label, d)


class BETA(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_beta(self.feature, self.label, self.periods)
        else:
            for d in self.periods:
                self.result["beta" + str(d)] = ts_beta(self.feature, self.label, d)


class REGRESSION(Alpha):
    def __init__(self, feature: pd.Series, label: pd.Series, periods: list[int] | int, rettype: int = 0,
                 normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.feature = feature
        self.label = label
        self.periods = periods
        self.rettype = rettype
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_regression(self.feature, self.label, self.periods, rettype=self.rettype)
        else:
            for d in self.periods:
                self.result["reg" + str(d)] = ts_regression(self.feature, self.label, d, self.rettype)


class PSY(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        diff = ts_delta(self.data, 1)
        if isinstance(self.periods, int):
            self.result = ts_pos_count(diff, self.periods) / self.periods * 100
        else:
            for d in self.periods:
                self.result["psy" + str(d)] = ts_pos_count(diff, d) / d * 100


class KBAR(Alpha):
    def __init__(self, data: pd.DataFrame, normalize: str = "zscore", nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.DataFrame(dtype='float64')

    def call(self):
        self.result["kmid"] = (self.data["close"] - self.data["open"]) / self.data["open"]
        self.result["klen"] = (self.data["high"] - self.data["low"]) / self.data["open"]
        self.result["kmid2"] = (self.data["close"] - self.data["open"]) / (self.data["high"] - self.data["low"])
        self.result["kup"] = (self.data["high"] - bigger(self.data["open"], self.data["close"])) / self.data["open"]
        self.result["kup2"] = (self.data["high"] - bigger(self.data["open"], self.data["close"])) / (
                self.data["high"] - self.data["low"])
        self.result["klow"] = (smaller(self.data["open"], self.data["close"]) - self.data["low"]) / self.data["open"]
        self.result["klow2"] = (smaller(self.data["open"], self.data["close"]) - self.data["low"]) / (
                self.data["high"] - self.data["low"])
        self.result["ksft"] = (2 * self.data["close"] - self.data["high"] - self.data["low"]) / self.data["open"]
        self.result["ksft2"] = (2 * self.data["close"] - self.data["high"] - self.data["low"]) / (
                self.data["high"] - self.data["low"])


class RSV(Alpha):
    def __init__(self, data: pd.DataFrame, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            lowest = ts_min(self.data["low"], self.periods)
            self.result = (self.data["close"] - lowest) / (ts_max(self.data["high"], self.periods) - lowest)
        else:
            for d in self.periods:
                lowest_d = ts_min(self.data["low"], d)
                self.result["rsv" + str(d)] = (self.data["close"] - lowest_d) / (
                        ts_max(self.data["high"], d) - lowest_d)


class CNTP(Alpha):
    # The percentage of days in past d days that price go up.
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        diff = ts_delta(self.data, 1)
        if isinstance(self.periods, int):
            self.result = ts_pos_count(diff, self.periods) / self.periods
        else:
            for d in self.periods:
                self.result["cntp" + str(d)] = ts_pos_count(diff, d) / d


class CNTN(Alpha):
    # The percentage of days in past d days that price go down.
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        diff = ts_delta(self.data, 1)
        if isinstance(self.periods, int):
            self.result = ts_neg_count(diff, self.periods) / self.periods
        else:
            for d in self.periods:
                self.result["cntn" + str(d)] = ts_neg_count(diff, d) / d


class SUMP(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        zeros = self.data - self.data
        diff = ts_delta(self.data, 1)
        if isinstance(self.periods, int):
            self.result = ts_sum(bigger(diff, zeros), self.periods) / ts_sum(abs(diff), self.periods)
        else:
            for d in self.periods:
                self.result["sump" + str(d)] = ts_sum(bigger(diff, zeros), d) / ts_sum(abs(diff), d)


class SUMN(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        zeros = self.data - self.data
        diff = ts_delta(self.data, 1)
        if isinstance(self.periods, int):
            self.result = ts_sum(bigger(-diff, zeros), self.periods) / ts_sum(abs(diff), self.periods)
        else:
            for d in self.periods:
                self.result["sumn" + str(d)] = ts_sum(bigger(-diff, zeros), d) / ts_sum(abs(diff), d)


class WQ_1(Alpha):
    # ts_decay_linear(ts_rank(close, 20) * cs_rank(volume) / cs_rank(returns), 15)
    def __init__(self, data: pd.DataFrame, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        c_rank = cs_rank(ts_returns(self.data["close"], 1))
        volume_rank = cs_rank(self.data["volume"])
        rank_ratio = volume_rank / c_rank
        if isinstance(self.periods, int):
            self.result = ts_decay(-ts_rank(self.data["close"], self.periods) * rank_ratio, 15)
        else:
            for d in self.periods:
                self.result["wq1_" + str(d)] = ts_decay(-ts_rank(self.data["close"], d) * rank_ratio, 15)


class WQ_2(Alpha):
    # cs_rank(ts_corr(returns, cs_mean(returns) * ts_decay_linear(close, 15), days))
    def __init__(self, data: pd.DataFrame, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.data = data
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        self.data["returns"] = ts_returns(self.data["close"], 1)
        self.data["cs_mean"] = cs_mean(self.data["returns"]) * ts_decay(self.data["close"], 15)
        if isinstance(self.periods, int):
            self.result = cs_rank(ts_corr(self.data["returns"], self.data["cs_mean"], self.periods))
        else:
            for d in self.periods:
                self.result["wq2_" + str(d)] = cs_rank(ts_corr(self.data["returns"], self.data["cs_mean"], d))


class CustomizedAlpha(Alpha):
    def __init__(self, data: pd.Series | pd.DataFrame, expression: list[str] | str, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        """
        eg:
        factor = CustomizedAlpha(data=df, expression=[f"ts_std(data['{x}'], 5)" for x in df.columns]).get_factor_value()
        """
        super().__init__()
        self.data = data
        self.expression = expression
        self.norma_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.expression, list):
            factors = []
            for e in self.expression:
                factors.append(eval(e.replace("data", "self.data")))
            self.result: pd.DataFrame = pd.concat(factors, axis=1)
        else:
            self.result: pd.Series = eval(self.expression.replace("data", "self.data"))


def qlib360(data: pd.DataFrame, normalize=False, fill=False, windows=None) -> pd.DataFrame:
    """
    复现qlib的alpha 360.
    将qlib源代码中的vwap替换成amount, 因为按照qlib的workflow, vwap全是空值, 则vwap类的因子是没有意义的

    :param data: 包括以下几列: open, close, high, low, volume, amount
    :param normalize: 是否进行cs zscore标准化
    :param fill: 是否向后填充缺失值
    :param windows: 列表, 默认为[5, 10, 20, 30, 60]
    :return:
    """
    if windows is None:
        windows = [i for i in range(59, 0, -1)]
    o_group = data["open"].groupby(level=1)
    c_group = data["close"].groupby(level=1)
    h_group = data["high"].groupby(level=1)
    l_group = data["low"].groupby(level=1)
    v_group = data["volume"].groupby(level=1)
    a_group = data["amount"].groupby(level=1)

    price = data["close"]
    volume = data["volume"]

    OPEN = DELAY(o_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    OPEN.columns = ["open" + str(w) for w in windows]
    for c in OPEN.columns:
        OPEN[c] /= price

    CLOSE = DELAY(c_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    CLOSE.columns = ["close" + str(w) for w in windows]
    for c in CLOSE.columns:
        CLOSE[c] /= price

    HIGH = DELAY(h_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    HIGH.columns = ["high" + str(w) for w in windows]
    for c in HIGH.columns:
        HIGH[c] /= price

    LOW = DELAY(l_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    LOW.columns = ["low" + str(w) for w in windows]
    for c in LOW.columns:
        LOW[c] /= price

    VOLUME = DELAY(v_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    VOLUME.columns = ["volume" + str(w) for w in windows]
    for c in VOLUME.columns:
        VOLUME[c] /= volume

    AMOUNT = DELAY(a_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    AMOUNT.columns = ["amount" + str(w) for w in windows]
    for c in AMOUNT.columns:
        AMOUNT[c] /= price * volume
    features = pd.concat([OPEN, CLOSE, HIGH, LOW, VOLUME, AMOUNT], axis=1)
    return features


def qlib158(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None,
            n_jobs: int = -1) -> pd.DataFrame:
    if windows is None:
        windows = [5, 10, 20, 30, 60]
    o_group = data["open"].groupby(level=1)
    c_group = data["close"].groupby(level=1)
    h_group = data["high"].groupby(level=1)
    l_group = data["low"].groupby(level=1)
    v_group = data["volume"].groupby(level=1)
    a_group = data["amount"].groupby(level=1)

    price = data["close"]
    volume = data["volume"]

    vol_mask = data["volume"].transform(lambda x: x if x > 0 else np.nan)

    tasks = [(KBAR(data).get_factor_value, (normalize, fill)),
             (BETA(data["open"], data["close"], windows).get_factor_value, (normalize, fill)),
             (RANK(c_group, windows).get_factor_value, (normalize, fill)),
             (RSV(data, windows).get_factor_value, (normalize, fill)),
             (CORR(data["close"], log(vol_mask), windows).get_factor_value, (normalize, fill)),
             (CORD(data["close"], data["volume"], windows).get_factor_value, (normalize, fill)),
             (CNTP(price, windows).get_factor_value, (normalize, fill)),
             (CNTN(price, windows).get_factor_value, (normalize, fill)),
             (SUMP(price, windows).get_factor_value, (normalize, fill)),
             (SUMN(price, windows).get_factor_value, (normalize, fill))]

    parallel_result = Parallel(n_jobs=n_jobs)(delayed(func)(*args) for func, args in tasks)
    parallel_df = pd.concat(parallel_result, axis=1)

    task2 = [(MA(c_group, windows).get_factor_value, (normalize, fill)),
             (STD(c_group, windows).get_factor_value, (normalize, fill)),
             (MAX(c_group, windows).get_factor_value, (normalize, fill)),
             (MIN(c_group, windows).get_factor_value, (normalize, fill)),
             (QTLU(c_group, windows).get_factor_value, (normalize, fill)),
             (QTLD(c_group, windows).get_factor_value, (normalize, fill))]
    parallel_result2 = Parallel(n_jobs=n_jobs)(delayed(func)(*args) for func, args in task2)
    parallel_df2 = pd.concat(parallel_result2, axis=1)
    for c in parallel_df2.columns:
        parallel_df2[c] /= price

    OPEN = DELAY(o_group, periods=[1, 2, 3, 4, 5]).get_factor_value(normalize=normalize, handle_nan=fill)
    OPEN.columns = ["open" + str(w) for w in range(1, 6)]
    for c in OPEN.columns:
        OPEN[c] /= price

    CLOSE = DELAY(c_group, periods=[1, 2, 3, 4, 5]).get_factor_value(normalize=normalize, handle_nan=fill)
    CLOSE.columns = ["close" + str(w) for w in range(1, 6)]
    for c in CLOSE.columns:
        CLOSE[c] /= price

    HIGH = DELAY(h_group, periods=[1, 2, 3, 4, 5]).get_factor_value(normalize=normalize, handle_nan=fill)
    HIGH.columns = ["high" + str(w) for w in range(1, 6)]
    for c in HIGH.columns:
        HIGH[c] /= price

    LOW = DELAY(l_group, periods=[1, 2, 3, 4, 5]).get_factor_value(normalize=normalize, handle_nan=fill)
    LOW.columns = ["low" + str(w) for w in range(1, 6)]
    for c in LOW.columns:
        LOW[c] /= price

    VOLUME = DELAY(v_group, periods=[1, 2, 3, 4, 5]).get_factor_value(normalize=normalize, handle_nan=fill)
    VOLUME.columns = ["volume" + str(w) for w in range(1, 6)]
    for c in VOLUME.columns:
        VOLUME[c] /= volume

    AMOUNT = DELAY(a_group, periods=[1, 2, 3, 4, 5]).get_factor_value(normalize=normalize, handle_nan=fill)
    AMOUNT.columns = ["amount" + str(w) for w in range(1, 6)]
    for c in AMOUNT.columns:
        AMOUNT[c] /= price * volume

    delta = DELTA(data["close"], periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    delta.columns = ["roc" + str(w) for w in windows]
    for c in delta.columns:
        delta[c] /= price

    r2 = REGRESSION(cs_rank(data["volume"]), cs_rank(data["close"]), windows, rettype=4).get_factor_value(
        normalize=normalize, handle_nan=fill)
    r2.columns = ["rsqr" + str(w) for w in windows]

    resi = REGRESSION(cs_rank(data["volume"]), cs_rank(data["close"]), windows, rettype=0).get_factor_value(
        normalize=normalize, handle_nan=fill)
    resi.columns = ["resi" + str(w) for w in windows]
    for c in resi.columns:
        resi[c] /= price

    vma = MA(v_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vma.columns = ["vma" + str(w) for w in windows]
    for c in vma.columns:
        vma[c] /= volume

    vstd = STD(v_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vstd.columns = ["vstd" + str(w) for w in windows]
    for c in vstd.columns:
        vstd[c] /= volume

    vsump = SUMP(data["volume"], windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vsump.columns = ["vsump" + str(w) for w in windows]

    vsumn = SUMN(data["volume"], windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vsumn.columns = ["vsumn" + str(w) for w in windows]

    features = pd.concat([OPEN, CLOSE, HIGH, LOW, VOLUME, AMOUNT, parallel_df, delta, r2,
                          resi, parallel_df2, vma, vstd, vsump, vsumn], axis=1)
    return features
