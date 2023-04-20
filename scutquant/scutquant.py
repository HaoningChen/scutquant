import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import math
import xgboost
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
import lightgbm as lgb
import pickle
import random
import warnings

warnings.filterwarnings("ignore")
random.seed(2046)


def join_data(data, data_join, time='datetime', col=None, index=None):
    """
    将序列数据(例如宏观的利率数据)按时间整合到面板数据中(例如沪深300成分)
    example:

    df_train = scutquant.join_data(df_train, series_train, col=['index_return', 'rf'], index=['datetime', instrument'])
    df_test = scutquant.join_data(df_test, series_test, col=['index_return', 'rf'], index=['datetime', 'instrument'])
    df = pd.concat([df_train, df_test], axis=0)

    :param data: pd.Series or pd.DataFrame, 股票数据(面板数据)
    :param data_join: pd.Series or pd.DataFrame, 要合并的序列数据
    :param time: str, 表示时间的列(两个数据集同时拥有)
    :param col: list, 被合并数据的列名(必须在data_join中存在)
    :param index: list, 面板数据的索引
    """
    if index is None:
        index = ['datetime', 'instrument']
    data = data.reset_index()
    data_join = data_join.reset_index()
    T = data[time].unique()

    asset_list = []
    for t in T:
        data_chosen = data[data[time] == t]  # 找出每天资产池中资产的数量
        asset_list.append(len(data_chosen))

    if col is not None:
        for c in col:
            idx = 0
            d_list = []
            for a in asset_list:
                data_join_chosen = data_join[c][idx]
                # print(data_join_chosen)
                for asset in range(a):
                    d_list.append(data_join_chosen)
                idx += 1
            data[c] = d_list
    data.set_index(index, inplace=True)
    if 'index' in data.columns:
        data = data.drop('index', axis=1)
    return data


def join_data_by_code(data, data_join, code='instrument', col=None, index=None):
    """
    场景：CAPM中每支股票对应一个beta和一个alpha（n*3的序列，包括股票代码、beta和alpha）, 将它们整合到股票的面板数据中（至少包括time, 股票代码）

    :param data: pd.DataFrame or pd.Series, 面板数据
    :param data_join: pd.DataFrame or pd.Series
    :param code: str, 表示股票代码的列
    :param col: list, data_join中待合并的列名
    :param index: list, 合并后设置的索引
    :return: pd.DataFrame
    """
    if index is None:
        index = ['datetime', 'instrument']
    data = data.reset_index()
    data_join = data_join.reset_index()
    for c in col:
        data[c] = 0
    for i in data[code].unique():
        ind = data[data[code] == i].index
        for c in range(len(col)):
            data_ = data_join[data_join[code] == i][col[c]].values[0]
            data.loc[ind, col[c]] = data_
    data.set_index(index, inplace=True)
    if 'index' in data.columns:
        data = data.drop('index', axis=1)
    return data


####################################################
# 特征工程
####################################################
def price2ret(price, shift1=-1, shift2=-2, groupby=None, fill=False):
    """
    return_rate = price_shift2 / price_shift1 - 1

    :param price: pd.DataFrame
    :param shift1: int, the value shift as denominator
    :param shift2: int, the value shift as numerator
    :param groupby: str
    :param fill: bool
    :return: pd.Series
    """
    if groupby is None:
        ret = price.shift(shift2) / price.shift(shift1).fillna(price.mean) - 1
    else:
        shift_1 = price.groupby([groupby]).shift(shift1)
        shift_2 = price.groupby([groupby]).shift(shift2)
        ret = shift_2 / shift_1 - 1
    if fill:
        ret.fillna(0, inplace=True)
    return ret


def zscorenorm(X, mean=None, std=None, clip=3):
    if mean is None:
        mean = X.mean()
    if std is None:
        std = X.std()
    X -= mean
    X /= std
    if clip is not None:
        X.clip(-clip, clip, inplace=True)
    return X


def robustzscorenorm(X, median=None, clip=3):
    if median is None:
        median = X.median()
    X -= median
    mad = abs(median) * 1.4826
    X /= mad
    if clip is not None:
        X.clip(-clip, clip, inplace=True)
    return X


def minmaxnorm(X, Min=None, Max=None, clip=3):
    if Min is None:
        Min = X.min()
    if Max is None:
        Max = X.max()
    X -= Min
    X /= Max - Min
    if clip is not None:
        X.clip(-clip, clip, inplace=True)
    return X


def ranknorm(X, groupby=None):
    if groupby is None:
        X_rank = X.rank(pct=True)
    else:
        X_rank = X.groupby(groupby).rank(pct=True)
    X_rank -= 0.5
    X_rank *= 3.46
    return X_rank


def make_pca(X):
    from sklearn.decomposition import PCA
    index = X.index
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=index)
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    result = {
        "pca": pca,
        "loadings": loadings,
        "X_pca": X_pca
    }
    return result


def plot_pca_variance(pca):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


def calc_multicollinearity(X, show=False):
    """
        反映多重共线性严重程度
    """
    corr = X.corr()
    if show:
        print(corr)
    corr = abs(corr)
    v = 0  # 此处是借用了vif的思想
    for c in corr.columns:
        if corr[c].mean() >= 0.6:
            v += 1
    return v / len(X.columns)


def make_mi_scores(X, y):
    """
    :param X: pd.DataFrame, 输入的特征
    :param y: pd.DataFrame or pd.Series, 输入的目标值
    :return: pd.Series, index为特征名，value为mutual information
    """
    from sklearn.feature_selection import mutual_info_regression
    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def make_r_scores(X, y):
    """
    :param X: pd.DataFrame or pd.Series, 特征值
    :param y: pd.DataFrame or pd.Series, 目标值
    :return: pd.Series, index为特征名, value为相关系数
    """
    r = []
    cols = X.columns
    for c in cols:
        r.append(pearson_corr(X[c], y))
    r = pd.Series(r, index=cols, name='R Scores').sort_values(ascending=False)
    return r


def show_dist(X):
    sns.kdeplot(X, shade=True)
    plt.show()


def feature_selector(df, score, value=0, verbose=0):
    """
    :param df: pd.DataFrame, 输入的数据(特征)
    :param score: pd.DataFrame, 特征得分，index为特征名，value为得分
    :param value: int or float, 筛选特征的临界值，默认为0
    :param verbose: bool, 是否输出被筛除的列
    :return: 被筛后的特征
    """
    col = score[score <= value].index
    df = df.drop(col, axis=1)
    if verbose == 1:
        for c in col:
            print(str(c) + ' will be dropped')
    return df


####################################################
# 数据清洗
####################################################
def align(x, y):
    """
    align x's index with y
    :param x: pd.DataFrame or pd.Series
    :param y: pd.DataFrame or pd.Series
    :return: pd.DataFrame(or pd.Series), pd.DataFrame(or pd.Series)
    """
    # print(x.index.names)
    # print(y.index.names)
    if len(x) > len(y):
        x = x[x.index.isin(y.index)]
    elif len(y) > len(x):
        y = y[y.index.isin(x.index)]
    return x, y


def percentage_missing(X):
    percent_missing = 100 * ((X.isnull().sum()).sum() / np.product(X.shape))
    return percent_missing


def process_inf(X: pd.DataFrame) -> pd.DataFrame:
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], X[col][~np.isinf(X[col])].mean())
    return X


def clean(X: pd.DataFrame) -> pd.DataFrame:
    X.dropna(axis=1, how='all', inplace=True)
    X.fillna(method='ffill', inplace=True)
    X.dropna(axis=0, inplace=True)
    # X = process_inf(X)
    return X


def calc_0(X, method='precise', val=0):  # 计算0或者其它数值的占比
    """
    :param X: pd.DataFrame, 输入的数据
    :param method: 'precise' or 'range'，需要计算占比的是数值还是某个范围
    :param val: int or float, 需要计算占比的具体数值
    :return: float, 比例
    """
    s = 0
    if method == 'precise':
        for i in range(0, len(X)):
            if X[i] == val:
                s += 1
    elif method == 'range':
        for i in range(0, len(X)):
            if -val <= X[i] <= val:
                s += 1
    return s / len(X)


def down_sample(X, col, val=0, n=0.35):
    """
    :param X: pd.DataFrame, 输入的数据
    :param col: str, 需要降采样的列名
    :param val: 需要降采样的样本值
    :param n: float, 降采样比例, 0~1
    :return: pd.DataFrame, 降采样后的数据集
    """
    X_0 = X[abs(X[col]) == val]
    n_drop = int(n * len(X_0))
    choice = np.random.choice(X_0.index, n_drop, replace=False)
    return X.drop(choice, axis=0)


def bootstrap(X, col, val=0, windows=5, n=0.35):
    """
    :param X: pd.DataFrame，输入的数据
    :param col: str, 需要升采样的列名
    :param val: 需要升采样的样本的值
    :param windows: int, 移动平均窗口，用来构建新样本
    :param n: float, 升采样比例，0~1
    :return: pd.DataFrame，扩充后的数据集
    """
    X_tar = X[X[col] == val]
    n_boot_drop = int(len(X_tar) * (1 - n))
    X_sample = pd.DataFrame(columns=X.columns, index=X_tar.index)
    for c in X_tar.columns:
        X_sample[c] = X_tar[c].rolling(window=windows, center=True, min_periods=int(0.5 * windows)).mean()
    choice = np.random.choice(X_sample.index, n_boot_drop, replace=False)
    X_sample = X_sample.drop(choice, axis=0)
    # print(X_sample)
    X = pd.concat((X, X_sample))
    return X


####################################################
# 拆分数据集
####################################################
def split_by_date(X, train_start_date, train_end_date, valid_start_date, valid_end_date):
    """
    :param X: pd.DataFrame
    :param train_start_date: str, 训练集的第一天, 例如“2020-12-28”
    :param train_end_date: str, 训练集最后一天
    :param valid_start_date: str, 验证集第一天, 例如"2022-12-28"
    :param valid_end_date: str, 验证集最后一天
    :return: pd.DataFrame, pd.DataFrame
    """
    X_train = X[X.index.get_level_values(0) <= train_end_date]
    X_train = X_train[X_train.index.get_level_values(0) >= train_start_date]
    X_valid = X[X.index.get_level_values(0) <= valid_end_date]
    X_valid = X_valid[X_valid.index.get_level_values(0) >= valid_start_date]
    return X_train, X_valid


def split(X, params=None):
    """
    相当于sklearn的train_test_split
    :param X: pd.DataFrame
    :param params: dict, 键名包括 "train", "valid", 值为比例
    :return: pd.DataFrame, pd.DataFrame
    """
    if params is None:
        params = {
            "train": 0.7,
            "valid": 0.3,
        }
    idx = X.index
    lis = [_ for _ in range(len(idx))]
    sample = random.sample(lis, int(len(lis) * params["valid"] + 0.5))
    idx_sample = idx[sample]
    X_valid = X[X.index.isin(idx_sample)]
    X_train = X[~X.index.isin(idx_sample)]
    return X_train, X_valid


def group_split(X, params=None):
    """
    以当天的所有股票为整体, 随机按比例拆出若干天作为训练集和验证集
    :param X: pd.DataFrame
    :param params: dict, 键名包括 "train", "valid", 值为比例
    :return: pd.DataFrame, pd.DataFrame
    """
    if params is None:
        params = {
            "train": 0.7,
            "valid": 0.3,
        }
    time = X.index.get_level_values(0).unique().values
    lis = [_ for _ in range(len(time))]
    sample = random.sample(lis, int(len(lis) * params["valid"] + 0.5))
    X_valid = X[X.index.get_level_values(0).isin(time[sample])]
    X_train = X[~X.index.isin(X_valid.index)]
    return X_train, X_valid


def split_data_by_date(data, kwargs):
    """
    按照日期拆出(整段)的测试集, 然后剩下的数据按照参数"split_method"和"split_kwargs"拆除训练集和验证机
    :param data: pd.DataFrame
    :param kwargs: dict, test_start_date必填, 其它选填. 当没指定test_end_date时, 默认截取到最后一天
    :return: pd.DataFrame
    """
    split_method = "split" if "split_method" not in kwargs.keys() else kwargs["split_method"]
    split_kwargs = None if "split_kwargs" not in kwargs.keys() else kwargs["split_kwargs"]

    test_start_date = kwargs["test_start_date"]  # 测试集的第一天
    dtest = data[data.index.get_level_values(0) >= test_start_date]
    # 默认测试集最后一天是数据集的最后一天
    if "test_end_date" in kwargs.keys():
        dtest = dtest[dtest.index.get_level_values(0) <= kwargs["test_end_date"]]
    dtrain = data[~data.index.isin(dtest.index)]

    if split_method == "split_by_date":
        # 默认训练集的第一天是数据集第一天，验证集的第一天是训练集最后一天的第二天
        if "train_start_date" not in split_kwargs.keys():
            train_start_date = dtrain.index.get_level_values(0)[0]
        else:
            train_start_date = split_kwargs["train_start_date"]
        if "train_start_date" not in split_kwargs.keys():
            valid_start_date = datetime.datetime.strptime(split_kwargs["train_end_date"], '%Y-%m-%d')
            valid_start_date += datetime.timedelta(days=1)
            valid_start_date = valid_start_date.strftime('%Y-%m-%d')
        else:
            valid_start_date = split_kwargs["valid_start_date"]
        dtrain, dvalid = split_by_date(dtrain, train_start_date, split_kwargs["train_end_date"], valid_start_date,
                                       split_kwargs["valid_end_date"])
    elif split_method == "split":
        dtrain, dvalid = split(dtrain, split_kwargs)
    else:
        dtrain, dvalid = group_split(dtrain, split_kwargs)
    return dtrain, dvalid, dtest


####################################################
# 自动处理器
####################################################
def auto_process(X, y, groupby=None, norm='z', label_norm=True, select=True, orth=True, clip=3, split_params=None):
    """
    :param X: pd.DataFrame，原始特征，包括了目标值
    :param y: str，目标值所在列的列名
    :param groupby: str, 如果是面板数据则输入groupby的依据，序列数据则直接填None
    :param norm: str, 标准化方式, 可选'z'/'r'/'m'
    :param label_norm: bool, 是否对目标值进行标准化
    :param select: bool, 是否去除无用特征
    :param orth: 是否正交化
    :param clip: 是否截断特征, None为不截断, 否则按照(-clip, clip)截断
    :param split_params: dict, 划分数据集的方法
    :return: dict{X_train, X_test, y_train, y_test, ymean, ystd}
    """
    date = X.index.names[0]
    if split_params is None:
        split_params = {
            "data": X,
            "test_date": None,
            "split_method": "group_split",
            "split_kwargs": {
                "train": 0.7,
                "valid": 0.3,
            }
        }

    print(X.info())
    X_mis = percentage_missing(X)
    print('X_mis=', X_mis)
    if groupby is None:
        X = clean(X)
    else:
        X.dropna(axis=1, how='all', inplace=True)
        X = X.groupby([groupby]).fillna(method='ffill').dropna()
    print('clean dataset done', '\n')

    # 拆分数据集
    X_train, X_valid, X_test = split_data_by_date(X, split_params)
    y_train, y_valid, y_test = X_train.pop(y), X_valid.pop(y), X_test.pop(y)

    X_train, y_train = align(X_train, y_train)
    X_valid, y_valid = align(X_valid, y_valid)
    X_test, y_test = align(X_test, y_test)

    print("split data done", "\n")

    # 降采样
    X_0 = calc_0(y_train)
    if X_0 > 0.5:
        print('The types of label value are imbalance, apply down sample method', '\n')
        X_train = down_sample(X_train, col=y)
        print('down sample done', '\n')

    # 目标值标准化
    if label_norm:
        if groupby is None:
            ymean, ystd = y_train.mean(), y_train.std()
            y_train, y_valid = zscorenorm(y_train, ymean, ystd), zscorenorm(y_valid, ymean, ystd)
        else:
            ymean, ystd = y_test.groupby(date).mean(), y_test.groupby(date).std()  # 是否应该使用滞后项
            y_train = zscorenorm(y_train, y_train.groupby(date).mean(), y_train.groupby(date).std())
            y_valid = zscorenorm(y_valid, y_valid.groupby(date).mean(), y_valid.groupby(date).std())
        print('label norm done', '\n')
    else:
        ymean, ystd = None, None
    print("The distribution of y_train:")
    show_dist(y_train)
    print("The distribution of y_valid:")
    show_dist(y_valid)
    print("The distribution of y_test:")
    show_dist(y_test)

    # 特征值标准化
    if groupby is None:
        if norm == 'z':
            mean, std = X_train.mean(), X_train.std()
            X_train = zscorenorm(X_train)
            X_valid, X_test = zscorenorm(X_valid, mean, std, clip), zscorenorm(X_test, mean, std, clip)
        elif norm == 'r':
            median = X_train.median()
            X_train = robustzscorenorm(X_train)
            X_valid, X_test = robustzscorenorm(X_valid, median, clip), robustzscorenorm(X_test, median, clip)
        elif norm == 'm':
            Min, Max = X_train.min(), X_train.max()
            X_train = minmaxnorm(X_train)
            X_valid, X_test = minmaxnorm(X_valid, Min, Max, clip), minmaxnorm(X_test, Min, Max, clip)
        else:
            X_train = ranknorm(X_train)
            X_valid, X_test = ranknorm(X_valid), ranknorm(X_test)
        X_train = clean(X_train)
        X_valid = clean(X_valid)
        X_test = clean(X_test)
    else:
        if norm == 'z':
            mean, std = X_train.groupby(date).mean(), X_train.groupby(date).std()
            X_train = zscorenorm(X_train, mean, std, clip)
            X_valid = zscorenorm(X_valid, X_valid.groupby(date).mean(), X_valid.groupby(date).std(), clip)
            X_test = zscorenorm(X_test, X_test.groupby(date).mean(), X_test.groupby(date).std(), clip)
        elif norm == 'r':
            median = X_train.groupby(date).median()
            X_train = robustzscorenorm(X_train, median, clip)
            X_valid = robustzscorenorm(X_valid, X_valid.groupby(date).median(), clip)
            X_test = robustzscorenorm(X_test, X_test.groupby(date).median(), clip)
        elif norm == 'm':
            Min, Max = X_train.groupby(date).min(), X_train.groupby(date).max()
            X_train = minmaxnorm(X_train, Min, Max, clip)
            X_valid = minmaxnorm(X_valid, X_valid.groupby(date).min(), X_valid.groupby(date).max(), clip)
            X_test = minmaxnorm(X_test, X_test.groupby(date).min(), X_test.groupby(date).max(), clip)
        else:
            X_train = ranknorm(X_train, groupby=date)
            X_valid = ranknorm(X_valid, groupby=date)
            X_test = ranknorm(X_test, groupby=date)

        X_train = X_train.groupby(groupby).fillna(method='ffill').dropna()
        X_valid = X_valid.groupby(groupby).fillna(method='ffill').dropna()
        X_test = X_test.groupby(groupby).fillna(method='ffill').dropna()

    print('norm data done', '\n')

    # PCA降维
    if orth:
        result = make_pca(X_train)
        pca, X_train = result["pca"], result["X_train"]
        X_valid, X_test = pca.transform(X_valid), pca.transform(X_test)

    # 特征选择
    if select:
        mi_score = make_mi_scores(X_train, y_train)
        print(mi_score)
        print(mi_score.describe())
        X_train = feature_selector(X_train, mi_score, value=0, verbose=1)
        X_valid = feature_selector(X_valid, mi_score)
        X_test = feature_selector(X_test, mi_score)

    X_train, y_train = align(X_train, y_train)
    X_valid, y_valid = align(X_valid, y_valid)
    X_test, y_test = align(X_test, y_test)
    print('all works done', '\n')
    returns = {
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "ymean": ymean,
        "ystd": ystd
    }
    return returns


####################################################
# 自动建模（线性回归模型）
####################################################
def auto_lrg(x, y, method='ols', fit_params=False, alphas=None, logspace_params=None, cv=10, max_iter=1000, verbose=1):
    """
    :param x: pd.DataFrame, 特征值
    :param y: pd.Series or pd.DataFrame, 目标值
    :param method: str, 回归方法, 可选'ols', 'lasso', 'ridge'和'logistic'
    :param fit_params: bool, 是否自动调参
    :param alphas: np.ndarray or others, 回归的超参数
    :param logspace_params: list[min, max, n_sample], 超参数搜索空间和采样的样本量
    :param cv: 参考sklearn的文档 'Determines the cross-validation splitting strategy.'
    :param max_iter: int, 最大迭代次数
    :param verbose: int, 等于1时输出使用的线性回归方法
    :return: model
    """
    if alphas is None:
        if logspace_params is None:
            logspace_params = [-5, 2, 200]
    from sklearn import linear_model
    model = None
    if verbose == 1:
        print(method + ' method will be used')
    if method == 'ols':
        lrg = linear_model.LinearRegression()
        model = lrg.fit(x, y)
    elif method == 'ridge':
        if fit_params:
            alphas = np.logspace(logspace_params[0], logspace_params[1], logspace_params[2])
            ridge_cv = linear_model.RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=cv)
            ridge_cv.fit(x, y)
            ridge = linear_model.Ridge(alpha=ridge_cv.alpha_, max_iter=max_iter)
        else:
            ridge = linear_model.Ridge()
        model = ridge.fit(x, y)
    elif method == 'lasso':
        if fit_params:
            alphas = np.logspace(logspace_params[0], logspace_params[1], logspace_params[2])
            lasso_cv = linear_model.LassoCV(alphas=alphas, cv=cv)
            lasso_cv.fit(x, y)
            lasso = linear_model.Lasso(alpha=lasso_cv.alpha_, max_iter=max_iter)
        else:
            lasso = linear_model.Lasso()
        model = lasso.fit(x, y)
    elif method == 'logistic':
        log = linear_model.LogisticRegression()
        model = log.fit(x, y)
    return model


class hybrid:
    def __init__(self, lin_model=None, xgb_model=None, task='reg', lrg_method='ols', alphas=None, logspace_params=None,
                 cv=10, max_iter=1000, xgb_params=None, weight=None):
        super(hybrid, self).__init__()
        self.task = task
        self.lrg_method = lrg_method
        self.alphas = alphas
        self.logspace_params = logspace_params
        self.cv = cv
        self.max_iter = max_iter
        self.xgb_params = xgb_params
        self.weight = weight
        self.lin_model = lin_model
        self.xgb_model = xgb_model

    def fit(self, x_train, y_train, x_valid, y_valid):
        if self.xgb_params is None:
            est = 800
            eta = 0.0421
            colsamp = 0.9325
            subsamp = 0.8785
            max_depth = 6
            l1 = 0.25
            l2 = 0.5
            early_stopping_rounds = 20
        else:
            est = self.xgb_params['est']
            eta = self.xgb_params['eta']
            colsamp = self.xgb_params['colsamp']
            subsamp = self.xgb_params['subsamp']
            max_depth = self.xgb_params['max_depth']
            l1 = self.xgb_params['l1']
            l2 = self.xgb_params['l2']
            early_stopping_rounds = self.xgb_params['early_stopping_rounds']
        if self.task == 'reg':
            xgb = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=est, eta=eta,
                                       colsample_bytree=colsamp, subsample=subsamp,
                                       reg_alpha=l1, reg_lambda=l2, max_depth=max_depth,
                                       early_stopping_rounds=early_stopping_rounds)
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        else:
            xgb = xgboost.XGBClassifier(n_estimators=est, eta=eta,
                                        colsample_bytree=colsamp, subsample=subsamp,
                                        reg_alpha=l1, reg_lambda=l2, max_depth=max_depth,
                                        early_stopping_rounds=early_stopping_rounds)
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        self.lin_model = auto_lrg(x_train, y_train, method=self.lrg_method, alphas=self.alphas,
                                  logspace_params=self.logspace_params, cv=self.cv, max_iter=self.max_iter)

    def predict(self, x_test):
        if self.weight is None:
            self.weight = [0.4, 0.6]
        pred_x = self.xgb_model.predict(x_test)
        pred_l = self.lin_model.predict(x_test)
        pred = []
        for i in range(0, len(pred_x)):
            pred.append(self.weight[0] * pred_l[i] + self.weight[1] * pred_x[i])
        # print(pred[0:5])
        return pred

    def save(self, target_dir):
        pickle.dump(self.lin_model, file=open(target_dir + '/linear.pkl', 'wb'))
        pickle.dump(self.xgb_model, file=open(target_dir + '/xgb.pkl', 'wb'))

    def load(self, target_dir):
        with open(target_dir + "/linear.pkl", "rb") as file:
            self.lin_model = pickle.load(file)
        file.close()
        with open(target_dir + "/xgb.pkl", "rb") as file:
            self.xgb_model = pickle.load(file)
        file.close()

    def explain_model(self, index):
        print('XGBoost Feature Importance:')
        xgboost.plot_importance(self.xgb_model)
        plt.show()
        importance = self.xgb_model.feature_importances_
        importance = pd.Series(importance, index=index).sort_values(ascending=False)
        print(importance, '\n')
        print('Linear Model Coef:')
        coef = self.lin_model.coef_
        c = pd.Series(coef, index=index).sort_values(ascending=False)
        print(c)


def auto_lgbm(x_train, y_train, x_valid, y_valid, early_stopping=30, verbose_eval=20, lgb_params=None,
              num_boost_round=1000, evals_result=None):
    if evals_result is None:
        evals_result = {}
    if lgb_params is None:
        lgb_params = {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
            "verbosity": -1
        }
    dtrain = lgb.Dataset(x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid)
    early_stopping_callback = lgb.early_stopping(early_stopping)
    verbose_eval_callback = lgb.log_evaluation(period=verbose_eval)
    evals_result_callback = lgb.record_evaluation(evals_result)
    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[early_stopping_callback, verbose_eval_callback, evals_result_callback],
    )
    return model


####################################################
# 评估指标
####################################################
def cov(x, y):
    x_bar = x.mean()
    y_bar = y.mean()
    cov_xy = 0
    for i in range(0, len(x)):
        cov_xy += (x[i] - x_bar) * (y[i] - y_bar)
    cov_xy = cov_xy / len(x)
    return cov_xy


def pearson_corr(x, y):
    np.array(x)
    np.array(y)
    x_std = x.std()
    y_std = y.std()
    cov_xy = cov(x, y)
    cor = cov_xy / (x_std * y_std)
    return cor


def ic_ana(pred, y, groupby=None, plot=True, freq=30):
    """
    :param pred: pd.DataFrame or pd.Series, 预测值
    :param y: pd.DataFrame or pd.Series, 真实值
    :param groupby: str, 排序依据
    :param plot: bool, 控制是否画出IC曲线
    :param freq: int, 频率, 用于平滑IC序列
    :return: float, 依次为ic均值, icir, rank_ic均值和rank_icir
    """
    groupby = pred.index.names[0] if groupby is None else groupby
    concat_data = pd.concat([pred, y], axis=1)
    ic = concat_data.groupby(groupby).apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1]))
    rank_ic = concat_data.groupby(groupby).apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], method='spearman'))
    if plot:
        # 默认freq为30的情况下，画出来的IC是月均IC
        plt.figure(figsize=(10, 6))
        plt.plot(ic.rolling(freq).mean(), label='ic', marker='o')
        plt.plot(rank_ic.rolling(freq).mean(), label='rank_ic', marker='o')
        plt.ylabel('score')
        plt.title('IC Series (rolling ' + str(freq) + ')')
        plt.legend()
        plt.show()
        plt.clf()
        show_dist(ic)
    IC, ICIR, Rank_IC, Rank_ICIR = ic.mean(), ic.mean() / ic.std(), rank_ic.mean(), rank_ic.mean() / rank_ic.std()
    return IC, ICIR, Rank_IC, Rank_ICIR


####################################################
# 时间序列分析
####################################################
def roll_mean(X, label, windows):
    """
    :param X: pd.DataFrame, 输入的数据
    :param label: str, 目标值的列名
    :param windows: int, 移动窗口
    :return: 增加了'moving_average'列的数据集
    """
    if windows % 2 == 0:
        min_periods = int(0.5 * windows)
    else:
        min_periods = int(0.5 * windows) + 1
    X['moving_average'] = X[label].rolling(
        window=windows,
        center=True,
        min_periods=min_periods
    ).mean()
    return X


def time_plot(X, label):
    """
    :param X: pd.DataFrame, 输入的数据
    :param label: str, 目标值所在列名
    :return: None（画图）
    """
    fig, ax = plt.subplots()
    X_copy = X.copy()
    X_copy['time_id'] = np.arange(len(X_copy))
    ax.plot('time_id', label, data=X_copy, color='0.75')
    ax = sns.regplot(x='time_id', y=label, data=X_copy, ci=None,
                     scatter_kws=dict(color='0.25'))
    ax.set_title('Time Plot of ' + label)
    plt.show()
    del X_copy


def series_plot(y, pred, fore, title=None, y_label='value', pred_label='pred', fore_label='fore'):
    """
    :param y: pd.DataFrame or pd.Series, 真实值
    :param pred: pd.DataFrame or pd.Series, 预测值（测试集）
    :param fore: pd.DataFrame or pd.Series, 预测值（训练集）
    :param title: str, 图的标题
    :param y_label: str, y的注释
    :param pred_label: str, pred的注释
    :param fore_label: str, fore的注释
    :return: None（画图）
    """
    ax = y.plot(alpha=0.5, title=title, ylabel=y_label)
    ax = pred.plot(ax=ax, linewidth=2, label=pred_label, color='C0')
    ax = fore.plot(ax=ax, linewidth=2, label=fore_label, color='C3')
    ax.legend()
    plt.show()


def make_trend(X, order, constant=False, drop_terms=True):
    from statsmodels.tsa.deterministic import DeterministicProcess
    dp = DeterministicProcess(
        index=X.index,  # dates from the training data
        constant=constant,  # dummy feature for the bias (y_intercept)
        order=order,  # the time dummy (trend)
        drop=drop_terms,  # drop terms if necessary to avoid collinearity
    )
    # `in_sample` creates features for the dates given in the `index` argument
    trend = dp.in_sample()
    return X.join(trend)


def plot_periodogram(X, time_freq='day', detrend='linear', ax=None):
    """
    The periodogram tells you the strength of the frequencies in a time series.

    :param X: pd.DataFrame or pd.Series, 输入的数据
    :param time_freq: str, 频率，目前只有'3sec', 'day', 'month' 和 'year'（未尝试，效果未知）
    :param detrend: str, 除趋势
    :param ax: 用于画图
    :return: None(画图)
    """
    if time_freq == 'day':
        fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    elif time_freq == 'month':
        fs = 12
    elif time_freq == '3sec':
        fs = 1200
    else:
        fs = 1
    freqencies, spectrum = periodogram(
        X,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    if time_freq == 'day':
        ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
        ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)",
                "Weekly (52)",
                "Semiweekly (104)",
            ],
            rotation=30,
        )
    elif time_freq == 'month':
        ax.set_xticks([1, 2, 4, 6, 12])
        ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
            ],
            rotation=15,
        )
    elif time_freq == '3sec':
        ax.set_xticks([1, 2, 4, 6, 12, 60, 120, 360, 720, 3600])
        ax.set_xticklabels(
            [
                "Hour (1)",
                "HalfanHour (2)",
                "Quarterly (4)",
                "10min (6)",
                "5min(12)",
                "1min(60)",
                "30sec(120)",
                "15sec(240)",
                "6sec(600)",
                "3sec(1200)"
            ],
            rotation=35,
        )
    else:
        ax.set_xticks([1])
        ax.set_xticklabels(
            [
                "Annual (1)",
            ],
            rotation=15,
        )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    plt.show()


def make_fourier_features(X, freq, order, name=None, time=None):
    """
    傅里叶特征：假设时间为t, 频率为f, 则特征 k = (2 * pi / f) * t

    :param X: pd.DataFrame, 输入的数据
    :param freq: int, 频率
    :param order: int, 阶数
    :param name: str, 自定义傅里叶特征的名字
    :param time: 表示时间的变量
    :return: pd.DataFrame, 加入了傅里叶特征的数据集
    """
    if time is None:
        time = np.arange(len(X.index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    if name is None:
        for i in range(1, order + 1):
            features.update({
                f"sin_{freq}_{i}": np.sin(i * k),
                f"cos_{freq}_{i}": np.cos(i * k),
            })
    else:
        for i in range(1, order + 1):
            features.update({
                f"{name}_sin_{freq}_{i}": np.sin(i * k),
                f"{name}_cos_{freq}_{i}": np.cos(i * k),
            })
    fourier = pd.DataFrame(features, index=X.index)
    return X.join(fourier)


def lagplot(x, y=None, lag=1, standardize=False, ax=None):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def lag_plot(x, lags, y=None, nrows=2, **kwargs):
    """
    :param x: pd.DataFrame, 输入的数据集，须包含目标值
    :param lags: int, 滞后的阶数
    :param y: str, 目标值所在列名
    :param nrows: int, 画图的行数
    :param kwargs: 其它参数
    :return: None（画图）
    """
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    plt.show()
    plot_pacf(x, lags=lags, method='ywm')
    plt.show()


def make_lags(X, data=None, lags=1, start=1, col=None, name=None):
    """
    :param X: pd.DataFrame, 整个数据集
    :param data: pd.Series or pd.DataFrame, 目标列, 不一定要在X中
    :param lags: int, 滞后阶数
    :param start: int, 从第几阶开始滞后
    :param col: list, 若需要滞后的变量不止一个, 即data为pd.DataFrame, 传入变量名的列表
    :param name: str, 为滞后项命名
    :return: pd.DataFrame, 加入了滞后项的数据集
    """
    if data is None:
        data = X
    if name is None:
        for i in range(start, lags + 1):
            if col is None:
                X[f'sin_{i}'] = data.shift(i)
            else:
                for c in col:
                    X[f'sin_{i}'] = data[c].shift(i)
    else:
        for i in range(start, lags + 1):
            if col is None:
                X[f'{name}_sin_{i}'] = data.shift(i)
            else:
                for c in col:
                    X[f'{name}_sin_{i}'] = data[c].shift(i)
    return X


def auto_ts_ana(X, label, freq, windows=5, lags=12):
    """
    :param X: pd.DataFrame, 包含目标值的整个数据集
    :param label: str, 目标值所在列名
    :param freq: str, 周期图的频率, 可选'3sec', 'day', 'month' 和 'year'
    :param windows: int, 移动平均窗口
    :param lags: int, 滞后项
    :return:
    """
    X_copy = X.copy()
    X_copy = roll_mean(X_copy, label, windows)
    time_plot(X_copy, label)
    plot_periodogram(X_copy[label], time_freq=freq)
    lag_plot(X_copy[label], lags)
    del X_copy
