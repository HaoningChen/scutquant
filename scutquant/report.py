import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def sharpe_ratio(ret, rf=0.03, freq=1):
    """
    夏普比率（事后夏普比率，使用实际收益计算）

    :param ret: pd.Series or pd.DataFrame, 收益率曲线
    :param rf: float, 无风险利率
    :param freq: int, 时间频率, 以年为单位则为1, 天为单位则为365, 其它类推
    :return: float, 对应时间频率的夏普比率
    """
    ret_copy = pd.Series(ret)
    rf /= freq
    return (ret_copy.mean() - rf) / ret_copy.std()


def information_ratio(ret, benchmark):
    """
    信息比率（与夏普比率的区别是使用指数收益作为对照的标准）

    :param ret: pd.Series or pd.DataFrame, 收益率曲线
    :param benchmark: pd.Series or pd.DataFrame, 基准回报率曲线
    :return: float
    """
    ret_copy = pd.Series(ret)
    benchmark_copy = pd.Series(benchmark)
    return (ret_copy.mean() - benchmark_copy.mean()) / ret_copy.std()


def calculate_mdd(data):
    """
    :param data: pd.Series
    :return: pd.Series
    """
    return data - data.cummax()


def plot(data, label, title=None, xlabel=None, ylabel=None, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    # plt.clf()
    for d in range(len(data)):
        plt.plot(data[d], label=label[d])
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def report_all(user_account, benchmark, ret=True, excess_return=True, risk=True, rf=0.03, freq=1):
    acc_val, ben_val = user_account.val_hist, benchmark.val_hist  # with cost
    init_val_acc = acc_val[0]
    init_val_ben = ben_val[0]

    acc_ret = []
    ben_ret = []
    for i in range(len(acc_val)):
        acc_ret.append(acc_val[i] / init_val_acc - 1)
        ben_ret.append(ben_val[i] / init_val_ben - 1)
    excess_ret = []
    for i in range(len(acc_ret)):
        excess_ret.append(acc_ret[i] - ben_ret[i])

    sharpe = sharpe_ratio(acc_ret, rf=rf, freq=freq)
    inf_ratio = information_ratio(acc_ret, ben_ret)
    acc_mdd = calculate_mdd(pd.Series(acc_ret))
    ben_mdd = calculate_mdd(pd.Series(ben_ret))
    print('Cumulative Rate of Return:', acc_ret[-1])
    print('Cumulative Rate of Return(benchmark):', ben_ret[-1])
    print('Cumulative Excess Rate of Return:', excess_ret[-1])
    print('Max Drawdown:', acc_mdd.min())
    print('Max Drawdown(benchmark):', ben_mdd.min())
    print('Sharpe Ratio:', sharpe)
    print('Information Ratio:', inf_ratio)

    if ret:
        plot([acc_ret, ben_ret], label=['cum_return_rate', 'benchmark'], title='Rate of Return',
             xlabel='time_id', ylabel='value')
    else:
        plot([acc_val, ben_val], label=['cum_return', 'benchmark'], title='Return', xlabel='time_id', ylabel='value')

    if excess_return:
        plot([excess_ret], label=['excess_return'], title='Excess Rate of Return', xlabel='time_id', ylabel='value')

    if risk:
        risk = pd.DataFrame({'risk': user_account.risk_curve})
        plot([risk], label=['risk_degree'], title='Risk Degree', xlabel='time_id', ylabel='value')


def group_return_ana(pred, n=5, groupby='time', figsize=(10, 6)):
    """
    因子对股票是否有良好的区分度, 若有, 则应出现明显的分层效应(即单调性)
    此处的收益为因子收益率，非真实收益率

    :param pred: pd.DataFrame or pd.Series, 预测值
    :param n: int, 分组数量(均匀地分成n组)
    :param groupby: str, groupby的索引
    :param figsize: 图片大小
    :return:
    """
    predict = pred.sort_values("predict", ascending=False)
    t_df = pd.DataFrame(
        {
            "Group%d"
            % (i + 1): predict.groupby(level=groupby)["predict"].apply(
                lambda x: x[len(x) // n * i: len(x) // n * (i + 1)].mean()  # 第 len(x) // n * i 行到 len(x) //n * (i+1) 行
            )
            for i in range(n)
        }
    )
    t_df.index = np.arange(len(t_df.index))
    # print(t_df.head(5))
    cols = t_df.columns
    data = []
    label = []
    for c in cols:
        data.append(t_df[c].cumsum())
        label.append(c)
    plot(data, label, title='Grouped Return', xlabel='time_id', ylabel='value', figsize=figsize)
