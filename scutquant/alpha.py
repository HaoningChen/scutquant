from .operators import *
import pandas as pd


def factor_neutralize(factors: pd.DataFrame, feature: list[str] | None, target: pd.Series) -> pd.DataFrame:
    return neutralize(factors, features=feature, target=target)


def market_neutralize(x: pd.Series, long_only: bool = False) -> pd.Series:
    """
    市场组合中性化:
    (1) 对所有股票减去其截面上的因子均值
    (2) 在(1)之后, 对每支股票除以截面上的因子值绝对值之和

    这样处理后每支股票会获得一个权重, 代表着资金的方向和数量(例如0.5代表半仓做多, -0.25代表1/4仓做空),
    且截面上的权重之和为0, 绝对值之和为1.
    """
    mean = x.groupby(level=0).mean()
    x -= mean
    if long_only:  # 考虑到A股有做空限制, 因此将权重为负的股票(即做空的股票)的权重调整为0(即纯多头), 并相应调整多头的权重
        x[x.values < 0] = 0
        abs_sum = x[x.values > 0].groupby(level=0).sum()
    else:
        abs_sum = abs(x).groupby(level=0).sum()
    x /= abs_sum
    return x


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
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class MA(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class STD(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class KURT(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class SKEW(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class DELAY(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class DELTA(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class MAX(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class MIN(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class RANK(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class QTLU(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class QTLD(Alpha):
    def __init__(self, data: pd.Series | pd.core.groupby.SeriesGroupBy, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class CORR(Alpha):
    def __init__(self, data: pd.DataFrame, feature: str, label: str, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
        super().__init__()
        self.data = data
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_corr(self.data, self.feature, self.label, self.periods)
        else:
            for d in self.periods:
                self.result["corr" + str(d)] = ts_corr(self.data, self.feature, self.label, d)

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class CORD(Alpha):
    # The correlation between feature change ratio and label change ratio
    def __init__(self, data: pd.DataFrame, feature: str, label: str, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
        super().__init__()
        self.data = data
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        self.data["fd"] = self.data[self.feature] / ts_delay(self.data[self.feature], 1)
        self.data["ld"] = self.data[self.label] / ts_delay(self.data[self.label], 1)
        if isinstance(self.periods, int):
            self.result = ts_corr(self.data, "fd", "ld", self.periods)
        else:
            for d in self.periods:
                self.result["cord" + str(d)] = ts_corr(self.data, "fd", "ld", d)

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class COV(Alpha):
    def __init__(self, data: pd.DataFrame, feature: str, label: str, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
        super().__init__()
        self.data = data
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_cov(self.data, self.feature, self.label, self.periods)
        else:
            for d in self.periods:
                self.result["cov" + str(d)] = ts_cov(self.data, self.feature, self.label, d)

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class BETA(Alpha):
    def __init__(self, data: pd.DataFrame, feature: str, label: str, periods: list[int] | int,
                 normalize: str = "zscore", nan_handling: bool = True):
        super().__init__()
        self.data = data
        self.feature = feature
        self.label = label
        self.periods = periods
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_beta(self.data, self.feature, self.label, self.periods)
        else:
            for d in self.periods:
                self.result["beta" + str(d)] = ts_beta(self.data, self.feature, self.label, d)

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class REGRESSION(Alpha):
    def __init__(self, data: pd.DataFrame, feature: str, label: str, periods: list[int] | int, rettype: int = 0,
                 normalize: str = "zscore", nan_handling: bool = True):
        super().__init__()
        self.data = data
        self.feature = feature
        self.label = label
        self.periods = periods
        self.rettype = rettype
        self.norm_method = normalize
        self.process_nan = nan_handling
        self.result = pd.Series(dtype='float64') | pd.DataFrame(dtype='float64')

    def call(self):
        if isinstance(self.periods, int):
            self.result = ts_regression(self.data, self.feature, self.label, self.periods, rettype=self.rettype)
        else:
            for d in self.periods:
                self.result["reg" + str(d)] = ts_regression(self.data, self.feature, self.label, d, self.rettype)

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class PSY(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class KBAR(Alpha):
    def __init__(self, data: pd.DataFrame, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class RSV(Alpha):
    def __init__(self, data: pd.DataFrame, periods: list[int] | int, normalize: str = "zscore",
                 nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class CNTP(Alpha):
    # The percentage of days in past d days that price go up.
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class CNTN(Alpha):
    # The percentage of days in past d days that price go down.
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class SUMP(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


class SUMN(Alpha):
    def __init__(self, data: pd.Series, periods: list[int] | int, normalize: str = "zscore", nan_handling: bool = True):
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

    def normalize(self):
        if self.norm_method == "zscore":
            self.result = cs_zscore(self.result)
        elif self.norm_method == "robust_zscore":
            self.result = cs_robust_zscore(self.result)
        elif self.norm_method == "scale":
            self.result = cs_scale(self.result)
        else:
            self.result = cs_rank(self.result)

    def handle_nan(self):
        if self.process_nan:
            self.result = self.result.groupby(level=1).transform(lambda x: x.fillna(method="ffill").fillna(0))

    def get_factor_value(self, normalize=False, handle_nan=False) -> pd.Series | pd.DataFrame:
        self.call()
        if normalize:
            self.normalize()
        if handle_nan:
            self.handle_nan()
        return self.result


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


def qlib158(data: pd.DataFrame, normalize: bool = False, fill: bool = False, windows=None) -> pd.DataFrame:
    """
    复现qlib的alpha 158.
    将一些指代不明的因子(例如rsqr, resi)和计算得不那么精确的因子(例如beta)替换成另外的表达式

    :param data: 包括以下几列: open, close, high, low, volume, amount
    :param normalize: 是否进行cs zscore标准化
    :param fill: 是否向后填充缺失值
    :param windows: 列表, 默认为[5, 10, 20, 30, 60]
    :return:
    """
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

    k = KBAR(data).get_factor_value(normalize=normalize, handle_nan=fill)

    delta = DELTA(data["close"], periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    delta.columns = ["roc" + str(w) for w in windows]
    for c in delta.columns:
        delta[c] /= price

    ma = MA(c_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    for c in ma.columns:
        ma[c] /= price

    std = STD(c_group, periods=windows).get_factor_value(normalize=normalize, handle_nan=fill)
    for c in std.columns:
        std[c] /= price

    beta = BETA(data, "open", "close", windows).get_factor_value(normalize=normalize, handle_nan=fill)

    r2 = REGRESSION(data, "open", "close", windows, rettype=4).get_factor_value(normalize=normalize, handle_nan=fill)
    r2.columns = ["rsqr" + str(w) for w in windows]

    resi = REGRESSION(data, "open", "close", windows, rettype=0).get_factor_value(normalize=normalize, handle_nan=fill)
    resi.columns = ["resi" + str(w) for w in windows]
    for c in resi.columns:
        resi[c] /= price

    cmax = MAX(c_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    for c in cmax.columns:
        cmax[c] /= price

    cmin = MIN(c_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    for c in cmin.columns:
        cmin[c] /= price

    qtlu = QTLU(c_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    for c in qtlu.columns:
        qtlu[c] /= price

    qtld = QTLD(c_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    for c in qtld.columns:
        qtld[c] /= price

    rank = RANK(c_group, windows).get_factor_value(normalize=normalize, handle_nan=fill)
    rsv = RSV(data, windows).get_factor_value(normalize=normalize, handle_nan=fill)

    vol_mask = data["volume"].transform(lambda x: x if x > 0 else np.nan)
    data["log_volume"] = log(vol_mask)
    corr = CORR(data, "close", "log_volume", windows).get_factor_value(normalize=normalize, handle_nan=fill)
    del data["log_volume"], vol_mask

    cord = CORD(data, "close", "volume", windows).get_factor_value(normalize=normalize, handle_nan=fill)
    cntp = CNTP(data["close"], windows).get_factor_value(normalize=normalize, handle_nan=fill)
    cntn = CNTN(data["close"], windows).get_factor_value(normalize=normalize, handle_nan=fill)
    sump = SUMP(data["close"], windows).get_factor_value(normalize=normalize, handle_nan=fill)
    sumn = SUMN(data["close"], windows).get_factor_value(normalize=normalize, handle_nan=fill)

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

    features = pd.concat(
        [OPEN, CLOSE, HIGH, LOW, VOLUME, AMOUNT, k, delta, ma, std, beta, r2, resi, cmax, cmin, qtlu, qtld, rank, rsv,
         corr, cord, cntp, cntn, sump, sumn, vma, vstd, vsump, vsumn],
        axis=1)
    return features
