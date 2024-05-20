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
    instrument_to = abs(factor_neu - ts_delay(factor_neu, 1).fillna(0))  # 今日权重 - 昨日权重，在单利回测情况下代表资金变动的百分比
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
    daily_return = X["factor_return"].groupby(level=0).sum()
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
        result["return"]: pd.Series = get_factor_portfolio(factor, label, long_only=long_only)
        result["return"] = pd.concat([pd.Series([1], index=[result["return"].index[0] - pd.DateOffset(days=1)]),
                                      result["return"]], axis=0)
        benchmark: pd.Series = label.groupby(level=0).mean()
        benchmark.index = pd.to_datetime(benchmark.index)
        benchmark = benchmark[benchmark.index.isin(result["return"].index)]
        result["benchmark"] = (benchmark + 1).cumprod()
        result["benchmark"] = pd.concat([pd.Series([1], index=[result["benchmark"].index[0] - pd.DateOffset(days=1)]),
                                         result["benchmark"]], axis=0)
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


def Brinson_Fachler_analysis(portfolio_weight: pd.Series, returns_benchmark: pd.Series, returns_portfolio=None,
                             benchmark_weight=None):
    """
    单期Brinson模型的BF分解. 可以简单累加作为多期Brinson模型的分解
    当returns_portfolio为None时，默认在benchmark成分内选股，即选择收益SR=0，超额收益完全来自配置收益AR
    当benchmark_weight为None时，benchmark默认为成分股等权指数
    为了正确计算中性组合的收益分解，对原式做了调整
    """
    if returns_portfolio is None:
        returns_portfolio: pd.Series = returns_benchmark
    # R_p: pd.Series = (portfolio_weight * returns_portfolio).groupby(level=0).sum()
    if benchmark_weight is None:
        benchmark_weight: pd.Series = returns_benchmark.groupby(level=0).transform(lambda x: 1 / x.count())
    R_b: pd.Series = (benchmark_weight * returns_benchmark).groupby(level=0).sum()

    adj_returns_portfolio = sign(portfolio_weight) * returns_portfolio
    adj_portfolio_weight = abs(portfolio_weight)

    # AR: pd.Series = ((portfolio_weight - benchmark_weight) * (returns_benchmark - R_b)).groupby(level=0).sum()
    # SR: pd.Series = R_p - R_b - AR
    SR: pd.Series = (benchmark_weight * (adj_returns_portfolio - returns_benchmark)).groupby(level=0).sum()
    AR: pd.Series = ((portfolio_weight - benchmark_weight) * (returns_benchmark - R_b)).groupby(level=0).sum()
    IR: pd.Series = ((adj_returns_portfolio - returns_benchmark) * (adj_portfolio_weight - benchmark_weight)).groupby(
        level=0).sum()
    return AR, SR, IR


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
        self.result["kmid"] = self.data["close"] / self.data["open"] - 1
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


class WVMA(Alpha):
    def __init__(self, price: pd.Series, volume: pd.Series, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: str = "ffill"):
        super().__init__()
        self.price = price
        self.volume = volume
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        weight: pd.Series = abs(ts_returns(self.price, 1))
        weighted_vol: pd.Series = weight * self.volume
        if isinstance(self.periods, int):
            self.result = ts_std(weighted_vol, self.periods) / ts_mean(weighted_vol, self.periods)
        else:
            for d in self.periods:
                self.result["wvma" + str(d)] = ts_std(weighted_vol, d) / ts_mean(weighted_vol, d)


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
    :param windows: 列表, 默认为[0-59]
    :return:
    """
    if windows is None:
        windows = [i for i in range(1, 60)]
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
    features = pd.concat([data[["open", "close", "high", "low", "volume", "amount"]], OPEN, CLOSE, HIGH, LOW, VOLUME,
                          AMOUNT], axis=1)
    return features


def qlib158(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None,
            n_jobs: int = -1, deunit: bool = True) -> pd.DataFrame:
    if windows is None:
        windows = [5, 10, 20, 30, 60]
    o_group = data["open"].groupby(level=1)
    c_group = data["close"].groupby(level=1)
    h_group = data["high"].groupby(level=1)
    l_group = data["low"].groupby(level=1)
    v_group = data["volume"].groupby(level=1)
    a_group = data["amount"].groupby(level=1)

    price = data["close"]
    volume = data["volume"].transform(lambda x: x if x > 0 else np.nan)

    tasks = [(KBAR(data).get_factor_value, (normalize, fill)),
             (BETA(data["open"], data["close"], windows).get_factor_value, (normalize, fill)),
             (RANK(c_group, windows).get_factor_value, (normalize, fill)),
             (RSV(data, windows).get_factor_value, (normalize, fill)),
             (CORR(data["close"], log(volume), windows).get_factor_value, (normalize, fill)),
             (CORD(data["close"], volume, windows).get_factor_value, (normalize, fill)),
             (CNTP(data["close"], windows).get_factor_value, (normalize, fill)),
             (CNTN(data["close"], windows).get_factor_value, (normalize, fill)),
             (SUMP(data["close"], windows).get_factor_value, (normalize, fill)),
             (SUMN(data["close"], windows).get_factor_value, (normalize, fill)),
             (WVMA(data["close"], volume, windows).get_factor_value, (normalize, fill))]

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

    if deunit:
        for c in parallel_df2.columns:
            parallel_df2[c] /= price

    OPEN = DELAY(o_group, periods=[1, 2, 3, 4]).get_factor_value(normalize=normalize, handle_nan=fill)
    OPEN.columns = ["open" + str(w) for w in range(1, 5)]

    CLOSE = DELAY(c_group, periods=[1, 2, 3, 4]).get_factor_value(normalize=normalize, handle_nan=fill)
    CLOSE.columns = ["close" + str(w) for w in range(1, 5)]

    HIGH = DELAY(h_group, periods=[1, 2, 3, 4]).get_factor_value(normalize=normalize, handle_nan=fill)
    HIGH.columns = ["high" + str(w) for w in range(1, 5)]

    LOW = DELAY(l_group, periods=[1, 2, 3, 4]).get_factor_value(normalize=normalize, handle_nan=fill)
    LOW.columns = ["low" + str(w) for w in range(1, 5)]

    VOLUME = DELAY(v_group, periods=[1, 2, 3, 4]).get_factor_value(normalize=normalize, handle_nan=fill)
    VOLUME.columns = ["volume" + str(w) for w in range(1, 5)]

    AMOUNT = DELAY(a_group, periods=[1, 2, 3, 4]).get_factor_value(normalize=normalize, handle_nan=fill)
    AMOUNT.columns = ["amount" + str(w) for w in range(1, 5)]

    basedata_price = pd.concat([OPEN, CLOSE, HIGH, LOW], axis=1)

    roc = DELTA(data["close"], periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    roc.columns = ["roc" + str(w) for w in windows]

    if deunit:
        for c in basedata_price.columns:
            basedata_price[c] /= price
        for c in VOLUME.columns:
            VOLUME[c] /= volume
        for c in AMOUNT.columns:
            AMOUNT[c] /= price * volume
        for c in roc.columns:
            roc[c] /= price

    r2 = REGRESSION(ts_returns(price, 1), ts_delay(ts_returns(price, 1), 1), windows, rettype=4).get_factor_value(
        normalize=normalize, handle_nan=fill)
    r2.columns = ["rsqr" + str(w) for w in windows]

    resi = REGRESSION(ts_returns(price, 1), ts_delay(ts_returns(price, 1), 1), windows, rettype=0).get_factor_value(
        normalize=normalize, handle_nan=fill)
    resi.columns = ["resi" + str(w) for w in windows]

    if deunit:
        for c in resi.columns:
            resi[c] /= price

    vma = MA(v_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vma.columns = ["vma" + str(w) for w in windows]

    if deunit:
        for c in vma.columns:
            vma[c] /= volume

    vstd = STD(v_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vstd.columns = ["vstd" + str(w) for w in windows]

    if deunit:
        for c in vstd.columns:
            vstd[c] /= volume

    vsump = SUMP(volume, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vsump.columns = ["vsump" + str(w) for w in windows]

    vsumn = SUMN(volume, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    vsumn.columns = ["vsumn" + str(w) for w in windows]

    features = pd.concat([data[["open", "close", "high", "low", "volume", "amount"]], basedata_price, VOLUME,
                          AMOUNT, parallel_df, roc, r2, resi, parallel_df2, vma, vstd, vsump, vsumn], axis=1)
    return features
