import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='Microsoft YaHei')


####################################################
# 特征工程
####################################################
def ZScoreNorm(X, mean=None, std=None, clip=False):
    if mean is None:
        mean = X.mean()
    if std is None:
        std = X.std()
    X -= mean
    X /= std
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def RobustZScoreNorm(X, median=None, clip=False):
    if median is None:
        median = X.median()
    X -= median
    mad = abs(median) * 1.4826
    X /= mad
    if clip:
        X.clip(-5, 5, inplace=True)
    return X


def MinMaxNorm(X, Min=None, Max=None, clip=False):
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
    X = X.fillna(method=method).fillna(0)
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


####################################################
# 拆分数据集
####################################################
def split(X, test_size=0.2):
    length = X.shape[0] - 1
    train_rows = int(length * (1 - test_size))
    X_train = X[0:train_rows].copy()
    X_test = X[train_rows - 1:-1].copy()
    return X_train, X_test


def sk_split(X, test_size, random_state=None):
    from sklearn.model_selection import train_test_split
    col = X.columns
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
    X_train = pd.DataFrame(X_train, columns=col)
    X_test = pd.DataFrame(X_test, columns=col)
    return X_train, X_test


def groupKfold_split(X, n_split):
    return X


####################################################
# 自动处理器
####################################################
def AutoProcessor(X, y, test_size=0.2, norm='z', describe=True, plot_x=False):
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
    print('clean dataset done')

    X_train, X_test = split(X, test_size=test_size)
    # print(X_train.shape, X_test.shape)
    X_0 = cal_0(X_train[y])
    if X_0 > 0.5:
        print('The types of label value are imbalance, apply down sample method')
        X_train = down_sample(X_train, col=y)
        print('down sample done')

    y_train, y_test = X_train.pop(y), X_test.pop(y)
    print('pop label done')
    print(y_train.describe())
    y_train = ZScoreNorm(y_train)
    show_dist(y_train)
    show_dist(y_test)

    if norm == 'z':
        mean, std = X_train.mean(), X_train.std()
        X_train = ZScoreNorm(X_train)
        X_test = ZScoreNorm(X_test, mean, std)
        print('norm data done')
    elif norm == 'r':
        median = X_train.median()
        X_train = RobustZScoreNorm(X_train)
        X_test = RobustZScoreNorm(X_test, median)
        print('norm data done')
    elif norm == 'm':
        Min, Max = X_train.min(), X_train.max()
        X_train = MinMaxNorm(X_train)
        X_test = MinMaxNorm(X_test, Min, Max)
        print('norm data done')

    if describe:
        print(X_train.describe())
        print(X_test.describe())

    if plot_x:
        for c in X.columns:
            show_dist(X[c])

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
    useless_fea = mi_score[mi_score.values == 0].index
    # print(useless_fea)
    X_train = X_train.drop(useless_fea, axis=1)
    X_test = X_test.drop(useless_fea, axis=1)
    # print(X_train.describe())
    print('all works done')
    return X_train, X_test, y_train, y_test
