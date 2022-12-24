import matplotlib.pyplot as plt
import pandas as pd


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


def plot(data, label, title=None, xlabel=None, ylabel=None):
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
        acc_ret.append(acc_val[i] / init_val_acc)
        ben_ret.append(ben_val[i] / init_val_ben)

    sharpe = sharpe_ratio(acc_ret, rf=rf, freq=freq)
    inf_ratio = information_ratio(acc_ret, ben_ret)
    print('Sharpe Ratio:', sharpe, 'Information Ratio:', inf_ratio)

    if ret:
        plot([acc_ret, ben_ret], label=['cum_return_rate', 'benchmark'], title='Rate of Return',
             xlabel='time_id', ylabel='value')
    else:
        plot([acc_val, ben_val], label=['cum_return', 'benchmark'], title='Return', xlabel='time_id', ylabel='value')

    if excess_return:
        excess_return = []
        for i in range(len(acc_ret)):
            excess_return.append(acc_ret[i] - ben_ret[i])
        plot([excess_return], label=['excess_return'], title='Excess Rate of Return', xlabel='time_id', ylabel='value')

    if risk:
        risk = pd.DataFrame({'risk': user_account.risk_curve})
        plot([risk], label=['risk_degree'], title='Risk Degree', xlabel='time_id', ylabel='value')
