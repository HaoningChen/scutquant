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


def sortino_ratio(ret, benchmark):
    """
    索提诺比率

    :param ret: pd.Series or pd.DataFrame, 收益率曲线
    :param benchmark: pd.Series or pd.DataFrame, 基准回报率曲线
    :return: float, 对应时间频率的夏普比率
    """
    ret = pd.Series(ret)
    benchmark_copy = pd.Series(benchmark)
    sd = ret[ret.values < benchmark_copy.values].std()
    return (ret.mean() - benchmark_copy.mean()) / sd


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


def plot(data, label, title=None, xlabel=None, ylabel=None, figsize=None, mode="plot"):
    """
    :param data: 需要绘制的数据
    :param label: 数据标签
    :param title: 标题
    :param xlabel: x轴的名字
    :param ylabel: y轴的名字
    :param figsize: 图片大小
    :param mode: 画图模式，有折线图”plot“和柱状图"bar"两种可供选择
    :return:
    """
    if figsize is not None:
        plt.figure(figsize=figsize)
    # plt.clf()
    if mode == "plot":
        for d in range(len(data)):
            plt.plot(data[d], label=label[d])
            plt.xticks(rotation=45)
    elif mode == "bar":
        bar = plt.bar(label, data, label="value")
        plt.bar_label(bar, label_type='edge')
        plt.xticks(rotation=45)
    else:
        raise ValueError("We don't support this mode: " + mode)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def accuracy(pred, y, sign=">="):
    """
    eg:
    y = pd.Series([-1, -1, 2, 3])
    y_hat = pd.Series([0, -2, 0, 2])
    data = y * y_hat
    label = 0
    sign = [">" for _ in range(len(data))]
    acc = accuracy(data, label, sign)  # A prediction is accurate if data[i] > label

    :param pred: pd.Series, 预测值
    :param y: pd.Series, 目标值
    :param sign: list, 运算符号, 长度与data相同
    :return: float, accuracy of prediction
    """
    data = pred * y
    # print(data)
    data_true = eval("data[data" + sign + "y]")
    return len(data_true) / len(data)


def report_all(user_account, benchmark, ret=True, excess_return=True, risk=True, rf=0.03, freq=1, time=None,
               figsize=(10, 6)):
    if time is not None:
        time = pd.to_datetime(time, format='%Y-%m-%d')

    acc_val, ben_val = user_account.val_hist, benchmark.val_hist  # with cost
    init_val_acc = acc_val[0]
    init_val_ben = ben_val[0]

    acc_ret = []
    ben_ret = []
    days = 0
    for i in range(len(acc_val)):
        acc_ret.append(acc_val[i] / init_val_acc - 1)
        ben_ret.append(ben_val[i] / init_val_ben - 1)
    excess_ret = []

    for i in range(len(acc_ret)):
        excess_ret.append(acc_ret[i] - ben_ret[i])
        if acc_ret[i] > 0:
            days += 1
    days /= len(acc_ret)

    sharpe = sharpe_ratio(acc_ret, rf=rf, freq=freq)
    sortino = sortino_ratio(acc_ret, ben_ret)
    inf_ratio = information_ratio(acc_ret, ben_ret)
    acc_mdd = calculate_mdd(pd.Series(acc_ret))
    ben_mdd = calculate_mdd(pd.Series(ben_ret))

    print('E(r):', pd.Series(acc_ret).mean())
    print('std:', pd.Series(acc_ret).std())
    print('E(r_benchmark):', pd.Series(ben_ret).mean())
    print('std_benchmark:', pd.Series(ben_ret).std(), '\n')
    print('Cumulative Rate of Return:', acc_ret[-1])
    print('Cumulative Rate of Return(benchmark):', ben_ret[-1])
    print('Cumulative Excess Rate of Return:', excess_ret[-1], '\n')
    print('Max Drawdown:', acc_mdd.min())
    print('Max Drawdown(benchmark):', ben_mdd.min(), '\n')
    print('Sharpe Ratio:', sharpe)
    print('Sortino Ratio:', sortino)
    print('Information Ratio:', inf_ratio, '\n')
    print('Pearson Correlation Coefficient Between Return and Benchmark:', pd.Series(acc_ret).corr(pd.Series(ben_ret)))
    print('Profitable Days(%):', days)

    if ret:
        acc_ret = pd.DataFrame(acc_ret, columns=["acc_ret"], index=time)
        ben_ret = pd.DataFrame(ben_ret, columns=["acc_ret"], index=time)
        plot([acc_ret, ben_ret], label=['cum_return_rate', 'benchmark'], title='Rate of Return',
             ylabel='value', figsize=figsize)
    else:
        acc_val = pd.DataFrame(acc_val, columns=["acc_val"], index=time)
        ben_val = pd.DataFrame(ben_val, columns=["acc_val"], index=time)
        plot([acc_val, ben_val], label=['cum_return', 'benchmark'], title='Return', ylabel='value',
             figsize=figsize)

    if excess_return:
        excess_ret = pd.DataFrame(excess_ret, columns=["excess_ret"], index=time)
        plot([excess_ret], label=['excess_return'], title='Excess Rate of Return', ylabel='value',
             figsize=figsize)

    if risk:
        risk = pd.DataFrame({'risk': user_account.risk_curve}, index=time)
        plot([risk], label=['risk_degree'], title='Risk Degree', ylabel='value', figsize=figsize)


def group_return_ana(pred, y_true, n=5, groupby='time', figsize=(10, 6)):
    """
    因子对股票是否有良好的区分度, 若有, 则应出现明显的分层效应(即单调性)
    此处的收益为因子收益率，非真实收益率

    :param pred: pd.DataFrame or pd.Series, 预测值
    :param y_true: pd.Series, 真实的收益率
    :param n: int, 分组数量(均匀地分成n组)
    :param groupby: str, groupby的索引
    :param figsize: 图片大小
    :return:
    """
    y_true.name = "label"
    y_true.index.names = pred.index.names
    if len(pred) > len(y_true):
        pred = pred[pred.index.isin(y_true.index)]
    else:
        y_true = y_true[y_true.index.isin(pred.index)]
    predict = pd.concat([pred, y_true], axis=1)
    # print(predict)
    predict = predict.sort_values("predict", ascending=False)
    acc = accuracy(predict["predict"], predict["label"], sign=">=")
    print('Accuracy of Prediction:', acc)
    t_df = pd.DataFrame(
        {
            "Group%d"
            % (i + 1): predict.groupby(level=groupby)["label"].apply(
                lambda x: x[len(x) // n * i: len(x) // n * (i + 1)].mean()  # 第 len(x) // n * i 行到 len(x) //n * (i+1) 行
            )
            for i in range(n)
        }
    )
    # 多空组合(即做多排名靠前的股票，做空排名靠后的股票)
    t_df["long-short"] = t_df["Group1"] - t_df["Group%d" % n]

    # Long-Average
    t_df["long-average"] = t_df["Group1"] - predict.groupby(level=groupby)["label"].mean()
    t_df.index = np.arange(len(t_df.index))
    # print(t_df.head(5))
    cols = t_df.columns
    data = []
    label = []
    win_rate = []
    for c in cols:
        data.append(t_df[c].cumsum())
        label.append(c)
        win_rate.append(len(t_df[t_df[c] >= 0]) / len(t_df))
    plot(data, label, title='Grouped Return', xlabel='time_id', ylabel='value', figsize=figsize)
    plot(win_rate, label=cols, title="Win Rate of Each Group", mode="bar", figsize=figsize)
