from . import scutquant as q
from . import alpha, models, executor, report
import yaml
import pandas as pd

"""
Build workflows with yaml files

将所有yaml文件和data文件放到同一目录, 例如 D:/Desktop/my_folder/
数据统一命名为data, 后缀支持csv, xls, xlsx, pkl, pickle, ftr, feather
所有结果也将保存到此目录
"""


def load_data(target_dir="", kwargs=None, auto_generate=False):
    if auto_generate and kwargs is None:
        kwargs = {
            "format": "csv",
            "encoding": "utf-8-sig",
            "index_col": ["datetime", "stock_code"],
            "index_names": None,
            "process_nan": True,
            "label": {
                "by": "close",
                "shift1": -1,
                "shift2": -2,
                "shift": -2,
                "add": 0,
                "divide": 1
            }
        }
    data_file = target_dir + "/data." + kwargs["format"]
    data_format = kwargs["format"]
    if data_format == "csv":
        data = pd.read_csv(data_file, encoding=kwargs["encoding"] if "encoding" in kwargs.keys() else None)
    elif data_format == "pkl" or data_format == "pickle":
        data = pd.read_pickle(data_file)
    elif data_format == "ftr" or data_format == "feather":
        data = pd.read_feather(data_file)
    elif data_format == "xls" or data_format == "xlsx":
        data = pd.read_excel(data_file)
    else:
        data = pd.DataFrame()
        print("Not support this format yet")
    index, index_names = kwargs["index_col"], kwargs["index_names"]
    if index is not None:
        data[index[0]] = pd.to_datetime(data[index[0]])  # 转换成日期格式
        data = data.set_index(index).sort_index()
    if index_names is not None:
        data.index.names = index_names
    data = data[~data.index.duplicated()]
    if kwargs["process_nan"]:
        data = data.groupby(data.index.names[1]).fillna(method="ffill").dropna()

    if "label" not in data.columns:
        if "add" not in kwargs["label"].keys():
            kwargs["label"]["add"] = 0
        if "divide" not in kwargs["label"].keys():
            kwargs["label"]["divide"] = 1
        # 计算label
        if "shift1" in kwargs["label"].keys() and "shift2" in kwargs["label"].keys():
            data["label"] = q.price2ret(data[kwargs["label"]["by"]], shift1=kwargs["label"]["shift1"],
                                        shift2=kwargs["label"]["shift2"], groupby=data.index.names[1])
        # 当除了"by"外没有任何信息时, 使用"by"并前移2项作为label
        else:
            if "shift" not in kwargs["label"].keys():
                kwargs["label"]["shift"] = -2
            data["label"] = data[kwargs["label"]["by"]].groupby(data.index.names[1]).shift(kwargs["label"]["shift"])
    data["label"] += kwargs["label"]["add"]
    data["label"] /= kwargs["label"]["divide"]
    return data


def get_factors(data, kwargs=None, auto_generate=False):
    if auto_generate and kwargs is None:
        kwargs = {
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "vol",
            "amount": "amount",
            "windows": [5, 10, 20, 30, 60],
            "fillna": False
        }
    kwargs["data"] = data
    X = alpha.make_factors(kwargs, windows=kwargs["windows"], fillna=kwargs["fillna"])
    return X


def concat_data(feature, label):
    label.dropna(inplace=True)
    feature = feature.groupby(feature.index.names[1]).fillna(method="ffill").dropna()
    feature, label = q.align(feature, label)
    data = pd.concat([feature, label], axis=1)
    return data


def process_data(data, kwargs=None, auto_generate=False):
    if auto_generate and kwargs is None:
        kwargs = {
            "label": "label",
            "groupby": "stock_code",
            "split": {
                "test_date": "2019-01-01",
                "split_method": "split",
                "split_kwargs": {
                    "train": 0.7,
                    "valid": 0.3
                }
            },
            "normalization": "z",
            "clip_data": 3,
            "label_norm": True,
            "select_features": False,
            "orth_data": False,
        }
    if kwargs["groupby"] is not None:
        kwargs["groupby"] = data.index.names[1]
    result = q.auto_process(data, kwargs["label"], groupby=kwargs["groupby"], norm=kwargs["normalization"],
                            label_norm=kwargs["label_norm"], select=kwargs["select_features"], orth=kwargs["orth_data"],
                            split_params=kwargs["split"])
    return result


def fit_data(result, target_dir="", kwargs=None, auto_generate=False):
    X_train, X_valid, X_test = result["X_train"], result["X_valid"], result["X_test"]
    y_train, y_valid, y_test = result["y_train"], result["y_valid"], result["y_test"]
    ymean, ystd = result["ymean"], result["ystd"]
    if kwargs is None and auto_generate:
        kwargs = {
            "model": "Ensemble",
            "params": {
                "epochs": 10,
            },
            "save": False
        }
    model = getattr(models, kwargs["model"])
    model = model(epochs=kwargs["params"]["epochs"])
    model.fit(X_train, y_train, X_valid, y_valid)
    if kwargs["save"]:
        model.save(target_dir)
    pred = model.predict(X_test)
    pred = pd.DataFrame(pred, columns=["predict"], index=X_test.index)
    pred["predict"] += ymean.groupby(ymean.index.names[0]).shift(2).fillna(0.0002)
    pred["predict"] *= ystd.groupby(ystd.index.names[0]).shift(2).fillna(0.0189)
    return pred


def analysis(predict, y_test, target_dir=""):
    time_periods = len(predict.index.get_level_values(0).unique())
    ic, icir, rank_ic, rank_icir = q.ic_ana(predict, y_test)
    print("IC Mean: ", ic, "ICIR:", icir, "Rank IC:", rank_ic, "Rank ICIR:", rank_icir)
    t_stat = icir * (time_periods ** 0.5)
    print("t-stat of alpha model:", t_stat)
    r = q.pearson_corr(predict["predict"], y_test)
    print("Pearson Correlation Coefficient:", r)
    result = pd.DataFrame({"ic": [ic], "icir": [icir], "rank_ic": [rank_ic], "rank_icir": [rank_icir],
                           "t-stat": [t_stat], "r": [r]})
    result.to_csv(target_dir + "/ic_results.csv")


def prepare_pred_data(predict, y_test, raw_data, kwargs=None, auto_generate=False):
    data = raw_data[raw_data.index.isin(y_test.index)]
    if auto_generate and kwargs is None:
        kwargs = {
            "deal_price": "close",
            "volume": "volume"
        }
    predict = executor.prepare(predict, data=data, price=kwargs["deal_price"], volume=kwargs["volume"], real_ret=y_test)
    return predict


def backtest(predict, kwargs=None, auto_generate=False):
    if auto_generate and kwargs is None:
        kwargs = {
            "generator": {
                "mode": "generate"
            },
            "strategy": {
                "class": "SigmaStrategy",
                "kwargs": {
                    "sigma": 1,
                    "auto_offset": False,
                    "offset_freq": 2,  # 应为delta_t + 1, 例如目标值是close_-2 / close_-1 - 1, 则delta_t = 1
                    "buy_only": False,  # =True时，只做多不做空(在A股做空有一定的难度)
                    "short_volume": 500,  # 融券做空的数量
                    "risk_degree": 0.95,  # 将风险度控制在这个数，如果超过了就按比例减持股票直到风险度小于等于它为止
                    "unit": None,  # 由于数据已经是以手为单位, 故无需二次处理
                    "max_volume": 0.05  # 手数随可用资金而改变，最大不会超过股票当天成交量的1%(例如T+1时下单，下单手数不会超过T时成交量的1%)
                }
            },
            "account": None,  # 使用默认账户, 即初始资金为1亿, 无底仓 (注意策略容量！)
            "trade_params": {
                "cost_buy": 0.0015,  # 佣金加上印花税
                "cost_sell": 0.0015,
                "min_cost": 5,
            }
        }
    generator, strategy, account, trade_params = \
        kwargs["generator"], kwargs["strategy"], kwargs["account"], kwargs["trade_params"]
    exe = executor.Executor(generator, strategy, account, trade_params)
    exe.execute(data=predict, verbose=0)  # verbose=1时，按时间输出买卖指令
    return exe


def report_results(exe, kwargs=None, auto_generate=False):
    account, benchmark = exe.user_account, exe.benchmark
    if kwargs is None and auto_generate:
        kwargs = {
            "freq": 1,
            "rf": 0.03,
        }
    rf, freq, time = kwargs["rf"], kwargs["freq"], exe.time
    report.report_all(account, benchmark, freq=freq, rf=rf, time=exe.time)


def pipeline(target_dir="", all_kwargs=None, auto_generate=True, save_prediction=True):
    if all_kwargs is None:
        with open(target_dir + "/all_kwargs.yaml", 'r') as file:
            all_kwargs = yaml.load(file, Loader=yaml.FullLoader)
        file.close()

    def init():
        if all_kwargs is not None:
            if "data" in all_kwargs.keys():
                data_kwargs = all_kwargs["data"]
            else:
                data_kwargs = None

            if "factor" in all_kwargs.keys():
                factor_kwargs = all_kwargs["factor"]
            else:
                factor_kwargs = None

            if "process" in all_kwargs.keys():
                process_kwargs = all_kwargs["process"]
            else:
                process_kwargs = None

            if "fit" in all_kwargs.keys():
                fit_kwargs = all_kwargs["fit"]
            else:
                fit_kwargs = None

            if "prepare" in all_kwargs.keys():
                prepare_kwargs = all_kwargs["prepare"]
            else:
                prepare_kwargs = None
            if "backtest" in all_kwargs.keys():
                backtest_kwargs = all_kwargs["backtest"]
            else:
                backtest_kwargs = None
            if "report" in all_kwargs.keys():
                report_kwargs = all_kwargs["report"]
            else:
                report_kwargs = None
        else:
            data_kwargs, factor_kwargs, process_kwargs, fit_kwargs, prepare_kwargs, backtest_kwargs, report_kwargs = \
                None, None, None, None, None, None, None
        return data_kwargs, factor_kwargs, process_kwargs, fit_kwargs, prepare_kwargs, backtest_kwargs, report_kwargs

    k_data, k_factor, k_process, k_fit, k_prepare, k_backtest, k_report = init()

    raw_data = load_data(target_dir=target_dir, kwargs=k_data, auto_generate=auto_generate)  # step1
    factor = get_factors(raw_data, k_factor, auto_generate=auto_generate)  # step2
    data_concat = concat_data(factor, raw_data["label"])  # step3
    result = process_data(data_concat, k_process, auto_generate=auto_generate)  # step4
    predict = fit_data(result, target_dir=target_dir, kwargs=k_fit, auto_generate=auto_generate)  # step5

    if save_prediction:
        predict.to_pickle(target_dir + "/predict.pkl")

    analysis(predict, result["y_test"], target_dir=target_dir)  # step6
    predict = prepare_pred_data(predict, result["y_test"], raw_data=raw_data, kwargs=k_prepare,
                                auto_generate=auto_generate)  # step7
    report.group_return_ana(predict, result["y_test"])  # step8
    exe = backtest(predict, kwargs=k_backtest, auto_generate=auto_generate)  # step9
    report_results(exe, kwargs=k_report, auto_generate=auto_generate)  # step10
