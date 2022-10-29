import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf


####################################################
# 特征工程
####################################################
def ZScoreNorm(X, mean=None, std=None, clip=True):
    if mean is None:
        mean = X.mean()
    if std is None:
        std = X.std()
    X -= mean
    X /= std
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def RobustZScoreNorm(X, median=None, clip=True):
    if median is None:
        median = X.median()
    X -= median
    mad = abs(median) * 1.4826
    X /= mad
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def MinMaxNorm(X, Min=None, Max=None, clip=True):
    if Min is None:
        Min = X.min()
    if Max is None:
        Max = X.max()
    X -= Min
    X /= Max - Min
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def make_pca(X):
    from sklearn.decomposition import PCA
    index = X.index
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=index)
    return X_pca


def symmetric(X):  # 对称正交
    col = X.columns
    index = X.index
    M = (X.shape[0] - 1) * np.cov(X.T.astype(float))
    D, U = np.linalg.eig(M)
    U = np.mat(U)
    d = np.mat(np.diag(D ** (-0.5)))
    S = U * d * U.T
    X = np.mat(X) * S
    X = pd.DataFrame(X, columns=col, index=index)
    return X


def cal_multicollinearity(X):
    """
        反映多重共线性严重程度
    """
    corr = X.corr()
    print(corr)
    corr = abs(corr)
    vif = 0
    for c in corr.columns:
        if corr[c].mean() >= 0.6:
            vif += 1
    return vif / len(X.columns)


def make_mi_scores(X, y):
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


def show_dist(X):
    sns.kdeplot(X, shade=True)
    plt.show()


def feature_selector(df, score, value=0):
    """
    :param df: dataframe like
    :param score: dataframe like
    :param value: default 0
    :return: df with useful features
    """
    col = score[score == value].index
    df = df.drop(col, axis=1)
    print(str(len(col)) + ' features will be dropped')
    return df


####################################################
# 数据清洗
####################################################
def percentage_missing(X):
    percent_missing = 100 * ((X.isnull().sum()).sum() / np.product(X.shape))
    return percent_missing


def dropna(X, axis=0):
    X = X.dropna(axis=axis)
    return X


def fillna(X, method='ffill'):
    X = X.fillna(method=method)
    return X


def process_inf(X):
    for col in X.columns:
        X[col] = X[col].replace([np.inf, -np.inf], X[col][~np.isinf(X[col])].mean())
    return X


def clean(X, axis=0):
    X = fillna(X)
    X = dropna(X, axis)
    return X


def cal_0(X, method='precise', val=0):  # 计算0或者其它数值的占比
    s = 0
    if method == 'precise':
        for i in range(0, len(X)):
            if X[i] == val:
                s += 1
    elif method == 'range':
        for i in range(0, len(X)):
            if abs(X[i]) <= val:
                s += 1
    return s / len(X)


def down_sample(X, col, val=0, n=0.35):
    X_0 = X[abs(X[col]) == val]
    n_drop = int(n * len(X_0))
    choice = np.random.choice(X_0.index, n_drop, replace=False)
    return X.drop(choice, axis=0)


def bootstrap(X, col, val=0, windows=5, n=0.35):
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
def split(X, test_size=0.2):
    length = X.shape[0] - 1
    train_rows = int(length * (1 - test_size))
    X_train = X[0:train_rows].copy()
    X_test = X[train_rows - 1:-1].copy()
    return X_train, X_test


def sk_split(X, y, test_size=0.2, random_state=None):
    from sklearn.model_selection import train_test_split
    x_col = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = pd.DataFrame(X_train, columns=x_col)
    X_test = pd.DataFrame(X_test, columns=x_col)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    return X_train, X_test, y_train, y_test


def groupKfold_split(X, y, n_split=5):
    from sklearn.model_selection import GroupKFold
    groups = np.arange(len(X))
    x_col = X.columns
    X = np.array(X)
    y = np.array(y)
    gkf = GroupKFold(n_splits=n_split)
    gkf.get_n_splits(X, y, groups)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    X_train = pd.DataFrame(X_train, columns=x_col)
    X_test = pd.DataFrame(X_test, columns=x_col)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    return X_train, X_test, y_train, y_test


####################################################
# 自动处理器
####################################################
def AutoProcessor(X, y, test_size=0.2, norm='z', label_norm=True, drop_useless_fea=True, orth=True,
                  describe=False, plot_x=False):
    """
        流程如下：
        初始化X，计算缺失值百分比，填充/去除缺失值，拆分数据集，判断训练集中目标值类别是否平衡并决定是否降采样（升采样和其它方法还没实现），分离Y并且画出其分布，
        对剩余的特征进行标准化，计算特征的相关系数判断是否存在多重共线性，如果存在则做PCA（因子正交的一种方法，也可以做对称正交，函数名为symmetric）
        完成以上工作后，计算X和Y的信息增益（mutual information，本质上是描述X可以解释多少的Y的一种指标），并去除MI Score为0（即无助于解释Y）的特征
    """
    print(X.info())
    X_mis = percentage_missing(X)
    print('X_mis=', X_mis)
    X = clean(X)
    print('clean dataset done', '\n')

    X_train, X_test = split(X, test_size=test_size)
    # print(X_train.shape, X_test.shape)
    X_0 = cal_0(X_train[y])
    if X_0 > 0.5:
        print('The types of label value are imbalance, apply down sample method', '\n')
        X_train = down_sample(X_train, col=y)
        print('down sample done', '\n')

    y_train, y_test = X_train.pop(y), X_test.pop(y)
    print('pop label done', '\n')
    # print(y_train.describe())
    if label_norm:
        ymean, ystd = y_train.mean(), y_train.std()
        y_train = ZScoreNorm(y_train)
        print('label norm done', '\n')
    else:
        ymean, ystd = 0, 1
    show_dist(y_train)
    show_dist(y_test)

    if norm == 'z':
        mean, std = X_train.mean(), X_train.std()
        X_train = ZScoreNorm(X_train)
        X_test = ZScoreNorm(X_test, mean, std)
    elif norm == 'r':
        median = X_train.median()
        X_train = RobustZScoreNorm(X_train)
        X_test = RobustZScoreNorm(X_test, median)
    elif norm == 'm':
        Min, Max = X_train.min(), X_train.max()
        X_train = MinMaxNorm(X_train)
        X_test = MinMaxNorm(X_test, Min, Max)

    X_train = clean(X_train)
    X_test = clean(X_test)
    print('norm data done', '\n')

    if describe:
        print(X_train.describe())
        print(X_test.describe())

    if plot_x:
        for c in X.columns:
            show_dist(X[c])

    if orth:
        r = cal_multicollinearity(X_train)
        if r > 0.35:
            print('To solve multicollinearity problem, orthogonal method will be applied')
            X_train = make_pca(X_train)
            # print(X_train.head(5))
            X_test = make_pca(X_test)
            # print(X_test.head(5))
            print('PCA done')

    mi_score = make_mi_scores(X_train, y_train)
    print(mi_score)
    print(mi_score.describe())
    if drop_useless_fea:
        feature_selector(X_train, mi_score)
        feature_selector(X_test, mi_score)
    # print(X_train.describe())
    print('all works done', '\n')
    return X_train, X_test, y_train, y_test, ymean, ystd


####################################################
# 自动建模（线性回归模型）
####################################################
def AutoLrg(x, y, method='ols', alphas=None, logspace_params=None, cv=10, max_iter=1000):
    if alphas is None:
        if logspace_params is None:
            logspace_params = [-5, 2, 200]
    from sklearn import linear_model
    model = None
    print(method + ' method will be used')
    if method == 'ols':
        lrg = linear_model.LinearRegression()
        model = lrg.fit(x, y)
    elif method == 'ridge':
        alphas = np.logspace(logspace_params[0], logspace_params[1], logspace_params[2])
        ridge_cv = linear_model.RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=cv)
        ridge_cv.fit(x, y)
        ridge = linear_model.Ridge(alpha=ridge_cv.alpha_, max_iter=max_iter)
        model = ridge.fit(x, y)
    elif method == 'lasso':
        alphas = np.logspace(logspace_params[0], logspace_params[1], logspace_params[2])
        ridge_cv = linear_model.LassoCV(alphas=alphas, cv=cv)
        ridge_cv.fit(x, y)
        ridge = linear_model.Lasso(alpha=ridge_cv.alpha_, max_iter=max_iter)
        model = ridge.fit(x, y)
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
        import xgboost
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
                                       reg_alpha=l1, reg_lambda=l2, max_depth=max_depth)
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
                                     early_stopping_rounds=early_stopping_rounds)
        else:
            xgb = xgboost.XGBClassifier(n_estimators=est, eta=eta,
                                        colsample_bytree=colsamp, subsample=subsamp,
                                        reg_alpha=l1, reg_lambda=l2, max_depth=max_depth)
            self.xgb_model = xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
                                     early_stopping_rounds=early_stopping_rounds)
        self.lin_model = AutoLrg(x_train, y_train, method=self.lrg_method, alphas=self.alphas,
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

    def dump(self, target_dir):
        import pickle
        pickle.dump(self.lin_model, file=open(target_dir + '/linear.pkl', 'wb'))
        pickle.dump(self.xgb_model, file=open(target_dir + '/xgb.pkl', 'wb'))


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


def ic_ana(pred, y, freq):
    from scipy import stats
    ic = []
    rank_ic = []
    for i in range(0, len(y), freq):
        sample_pred = pred[i:i + freq]
        sample_y = y[i:i + freq]
        ic.append(pearson_corr(sample_pred, sample_y))
        rank_ic.append(stats.spearmanr(sample_pred, sample_y)[0])
    ic = pd.Series(ic)
    rank_ic = pd.Series(rank_ic)
    plt.plot(ic, label='ic', marker='o')
    plt.plot(rank_ic, label='rank_ic', marker='o')
    plt.xlabel('time_id')
    plt.ylabel('score')
    plt.title('IC Series')
    plt.legend()
    plt.show()
    IC, ICIR, Rank_IC, Rank_ICIR = ic.mean(), ic.mean() / ic.std(), rank_ic.mean(), rank_ic.mean() / rank_ic.std()
    return IC, ICIR, Rank_IC, Rank_ICIR


####################################################
# 时间序列分析
####################################################
def roll_mean(X, label, windows):
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
            rotation=15,
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


def make_fourier_features(X, freq, order):
    time = np.arange(len(X.index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
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


def make_lags(X, data=None, lags=1, col=None):
    if data is None:
        data = X
    for i in range(1, lags + 1):
        if col is None:
            X['lag_' + str(i)] = data.shift(i)
        else:
            for c in col:
                X[c + '_' + str(i)] = data[c].shift(i)
    return X


def Auto_ts_ana(X, label, freq, windows=5, lags=12):
    X_copy = X.copy()
    X_copy = roll_mean(X_copy, label, windows)
    time_plot(X_copy, label)
    plot_periodogram(X_copy[label], time_freq=freq)
    lag_plot(X_copy[label], lags)
    del X_copy
