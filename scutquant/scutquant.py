import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import xgboost
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_pacf
import lightgbm as lgb


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
        data_choosed = data[data[time] == t]  # 找出每天资产池中资产的数量
        asset_list.append(len(data_choosed))

    if col is not None:
        for c in col:
            idx = 0
            d_list = []
            for a in asset_list:
                data_join_choosed = data_join[c][idx]
                # print(data_join_choosed)
                for asset in range(a):
                    d_list.append(data_join_choosed)
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
def price2ret(price, shift1=-1, shift2=-21, groupby=None, fillna=False):
    """
    return_rate = price_shift2 / price_shift1 - 1

    :param price: pd.DataFrame
    :param shift1: int
    :param shift2: int
    :param groupby: str
    :param fillna: bool
    :return: pd.Series
    """
    if groupby is None:
        ret = price.shift(shift2) / price.shift(shift1) - 1
    else:
        ret = price.groupby([groupby]).shift(shift2) / price.groupby([groupby]).shift(shift1) - 1
    if fillna:
        ret.fillna(0, inplace=True)
    return ret


def zscorenorm(X, mean=None, std=None, clip=True):
    if mean is None:
        mean = X.mean()
    if std is None:
        std = X.std()
    X -= mean
    X /= std
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def robustzscorenorm(X, median=None, clip=True):
    if median is None:
        median = X.median()
    X -= median
    mad = abs(median) * 1.4826
    X /= mad
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def minmaxnorm(X, Min=None, Max=None, clip=True):
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


def cal_multicollinearity(X, show=False):
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
    X.dropna(axis=1, how='all', inplace=True)
    X = fillna(X)
    X = dropna(X, axis)
    return X


def cal_0(X, method='precise', val=0):  # 计算0或者其它数值的占比
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


def groupkfold_split(X, y, n_split=5):
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
def auto_process(X, y, test_size=0.2, groupby=None, norm='z', label_norm=True, select=True, orth=True,
                 describe=False, plot_x=False):
    """
    流程如下：
    初始化X，计算缺失值百分比，填充/去除缺失值，拆分数据集，判断训练集中目标值类别是否平衡并决定是否降采样（升采样和其它方法还没实现），分离Y并且画出其分布，
    对剩余的特征进行标准化，计算特征的相关系数判断是否存在多重共线性，如果存在则做PCA（因子正交的一种方法，也可以做对称正交，函数名为symmetric）
    完成以上工作后，计算X和Y的信息增益（mutual information，本质上是描述X可以解释多少的Y的一种指标），并去除MI Score为0（即无助于解释Y）的特征

    An example:

    data['label'] = make_label(data)
    data = make_features(data)
    feature_train, feature_test, label_train, label_test = auto_process(data, 'label', test_size=0.25, norm='r')

    :param X: pd.DataFrame，原始特征，包括了目标值
    :param y: str，目标值所在列的列名
    :param test_size: float, 测试集占数据集的比例
    :param groupby: str, 如果是面板数据则输入groupby的依据，序列数据则直接填None
    :param norm: str, 标准化方式, 可选'z'/'r'/'m'
    :param label_norm: bool, 是否对目标值进行标准化
    :param select: bool, 是否去除无用特征
    :param orth: 是否正交化
    :param describe: bool, 是否输出处理好的特征的前描述统计
    :param plot_x: bool, 是否画出X的分布
    :return: X_train, X_test, y_train, y_test, ymean, ystd
    """
    print(X.info())
    X_mis = percentage_missing(X)
    print('X_mis=', X_mis)
    if groupby is None:
        X = clean(X)
    else:
        X.dropna(axis=1, how='all', inplace=True)
        X = X.groupby([groupby]).fillna(method='ffill').dropna()
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
        y_train = zscorenorm(y_train)
        print('label norm done', '\n')
    else:
        ymean, ystd = 0, 1
    show_dist(y_train)
    show_dist(y_test)

    if select:
        mi_score = make_mi_scores(X_train, y_train)
        print(mi_score)
        print(mi_score.describe())
        X_train = feature_selector(X_train, mi_score, value=0, verbose=1)
        X_test = feature_selector(X_test, mi_score)

    if norm == 'z':
        mean, std = X_train.mean(), X_train.std()
        X_train = zscorenorm(X_train)
        X_test = zscorenorm(X_test, mean, std)
    elif norm == 'r':
        median = X_train.median()
        X_train = robustzscorenorm(X_train)
        X_test = robustzscorenorm(X_test, median)
    elif norm == 'm':
        Min, Max = X_train.min(), X_train.max()
        X_train = minmaxnorm(X_train)
        X_test = minmaxnorm(X_test, Min, Max)
    if groupby is None:
        X_train = clean(X_train)
        X_test = clean(X_test)
    else:
        X_train.dropna(axis=1, how='all', inplace=True)
        X_test.dropna(axis=1, how='all', inplace=True)
        X_train = X_train.groupby([groupby]).fillna(method='ffill').dropna()
        X_test = X_test.groupby([groupby]).fillna(method='ffill').dropna()
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
    # print(X_train.describe())
    print('all works done', '\n')
    return X_train, X_test, y_train, y_test, ymean, ystd


####################################################
# 自动建模（线性回归模型）
####################################################
def auto_lrg(x, y, method='ols', fit_params=False, alphas=None, logspace_params=None, cv=10, max_iter=1000):
    """
    :param x: pd.DataFrame, 特征值
    :param y: pd.Series or pd.DataFrame, 目标值
    :param method: str, 回归方法, 可选'ols', 'lasso', 'ridge'和'logistic'
    :param fit_params: bool, 是否自动调参
    :param alphas: np.ndarray or others, 回归的超参数
    :param logspace_params: list[min, max, n_sample], 超参数搜索空间和采样的样本量
    :param cv: 参考sklearn的文档 'Determines the cross-validation splitting strategy.'
    :param max_iter: int, 最大迭代次数
    :return: model
    """
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

    def dump(self, target_dir):
        import pickle
        pickle.dump(self.lin_model, file=open(target_dir + '/linear.pkl', 'wb'))
        pickle.dump(self.xgb_model, file=open(target_dir + '/xgb.pkl', 'wb'))

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


def ic_ana(pred, y, groupby=None, freq=1, plot=True):
    """
    :param pred: pd.DataFrame or pd.Series, 预测值
    :param y: pd.DataFrame or pd.Series, 真实值
    :param groupby: str, 排序依据
    :param freq: int, 仅在groupby=None下生效，作用是对个资产返回一段时间（调仓频率）的pearson相关系数
    :param plot: bool, 控制是否画出IC曲线
    :return: float, 依次为ic均值, icir, rank_ic均值和rank_icir
    """
    if groupby is not None:
        IC = pd.concat([pred, y], axis=1).groupby([groupby])
        ic = IC.corr().iloc[:, 0]
        ic = ic[[i % 2 == 1 for i in range(len(ic.index))]]
        # print('ic:', ic)
        rank_ic = IC.corr(method='spearman').iloc[:, 0]
        rank_ic = rank_ic[[i % 2 == 1 for i in range(len(rank_ic.index))]]
    else:
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
    # print('rank_ic:', rank_ic)
    if plot:
        plt.plot(ic.values, label='ic', marker='o')
        plt.plot(rank_ic.values, label='rank_ic', marker='o')
        plt.xlabel(groupby + '_id')
        plt.ylabel('score')
        plt.title('IC Series')
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


def make_fourier_features(X, freq, order, name=None):
    """
    傅里叶特征：假设时间为t, 频率为f, 则特征 k = (2 * pi / f) * t

    :param X: pd.DataFrame, 输入的数据
    :param freq: int, 频率
    :param order: int, 阶数
    :param name: str, 自定义傅里叶特征的名字
    :return: pd.DataFrame, 加入了傅里叶特征的数据集
    """
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
    :return: None(画图)
    """
    X_copy = X.copy()
    X_copy = roll_mean(X_copy, label, windows)
    time_plot(X_copy, label)
    plot_periodogram(X_copy[label], time_freq=freq)
    lag_plot(X_copy[label], lags)
    del X_copy
