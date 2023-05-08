import akshare as ak
import pandas as pd
import datetime

# from joblib import Parallel, delayed

"""
akshare的数据并非100%准确！如果有更好的数据源请使用自己的数据
不知为何sh000001和sh000002有问题
"""


def get_index_stock_cons(index_code='000300', freq="daily", start="20230330", end="20230331", adjust=""):
    """
    注：此函数还在不断完善中, 尤其是股票代码一块，非沪深300股票池的股票, 代码后缀可能会出错
    :param index_code: str, 指数代码
    :param freq: str, 有"daily", "weekly"和"monthly"可选
    :param start: str, 日期, %y%m%d格式
    :param end: str, 日期, %y%m%d格式
    :param adjust: ""为不复权, “qfq”为前复权, “hfq”为后复权
    :return: pd.DataFrame
    example:
    data = get_index_stock_cons()
    """
    cons = ak.index_stock_cons(symbol=index_code)
    df = pd.DataFrame()
    for code in cons["品种代码"]:
        stock_data = ak.stock_zh_a_hist(symbol=code, period=freq, start_date=start, end_date=end, adjust=adjust)
        stock_data["code"] = code + ".SH" if code[0] == "6" else code + ".SZ"  # 根据股票代码的第一个数字区分其属于上交所还是深交所
        df = pd.concat([df, stock_data], axis=0)
    df = df.set_index(["日期", "code"]).sort_index()
    df.index.names = ["datetime", "code"]
    df = df[~df.index.duplicated()]
    df.columns = ["open", "close", "high", "low", "volume", "amount", "amplitude", "price_chg", "pcg_chg", "turnover"]
    return df


def upgrade_index_stock_cons(index_code='000300', today=None, adjust=""):
    """
    此函数设计的目的是自动更新数据
    :param index_code: str, 指数代码
    :param today: str, 今天的日期, %y%m%d格式
    :param adjust: ""为不复权, “qfq”为前复权, “hfq”为后复权
    :return: pd.DataFrame
    example:
    data = upgrade_index_stock_cons(today="20230330")
    """
    if today is None:
        today = datetime.date.today()
        today = today.strftime("%Y%m%d")
    df = get_index_stock_cons(index_code=index_code, freq="daily", start=today, end=today, adjust=adjust)
    return df


def get_daily_data(index_code, adjust=""):
    """
    获取指数成分股的历史数据(动态股票池, 日频), 支持各种复权
    一次性获取所有日期的数据

    :param index_code: 指数代码, like "sh000300"
    :param adjust: ""为不复权, “qfq”为前复权, “hfq”为后复权
    :return: pd.DataFrame
    """
    all_stocks = ak.index_stock_hist(symbol=index_code)
    all_stocks["in_date"] = pd.to_datetime(all_stocks["in_date"]).dt.strftime('%Y%m%d')
    all_stocks["out_date"] = pd.to_datetime(all_stocks["out_date"]).dt.strftime('%Y%m%d')

    data = pd.DataFrame()

    for stock in all_stocks["stock_code"].unique():
        start, end = all_stocks[all_stocks["stock_code"] == stock]["in_date"].unique(), \
            all_stocks[all_stocks["stock_code"] == stock]["out_date"].unique()
        # print(start, end)
        for i in range(len(start)):
            stock_data = ak.stock_zh_a_hist(symbol=stock, period="daily", start_date=start[i], end_date=end[i],
                                            adjust=adjust)
            stock_data["code"] = stock + ".SH" if stock[0] == "6" else stock + ".SZ"
            data = pd.concat([data, stock_data], axis=0)
    data = data.set_index(["日期", "code"]).sort_index()
    data.index.names = ["datetime", "code"]
    data = data[~data.index.duplicated()]
    data.columns = ["open", "close", "high", "low", "volume", "amount", "amplitude", "price_chg", "pct_chg", "turnover"]
    return data


"""
def get_high_freq_data(index_code="000300", minutes=1, adjust="hfq"):
    def get_minute_data(code, minute, adj):
        stock_code = "sh" + code if code[0] == "6" else "sz" + code
        stock_data = ak.stock_zh_a_minute(symbol=stock_code, period=str(minute), adjust=adj)
        stock_data["code"] = stock_code
        stock_data.set_index(["day", "code"], inplace=True)
        return stock_data

    cons = ak.index_stock_cons(symbol=index_code)
    df_list = Parallel(n_jobs=-1)(delayed(get_minute_data)(code, minutes, adjust) for code in cons["品种代码"])
    df = pd.concat(df_list, axis=0)
    df = df[~df.index.duplicated()]
    return df
"""


def get_high_freq_data(index_code="000300", minutes=1, adjust="hfq"):
    df = pd.DataFrame()
    cons = ak.index_stock_cons(symbol=index_code)
    for code in cons["品种代码"]:
        stock_code = "sh" + code if code[0] == "6" else "sz" + code
        stock_data = ak.stock_zh_a_minute(symbol=stock_code, period=str(minutes), adjust=adjust)
        stock_data["code"] = stock_code
        df = pd.concat([df, stock_data], axis=0)
    df = df.set_index(["day", "code"]).sort_index()
    df.dropna(axis=1, how='all', inplace=True)
    df = df[~df.index.duplicated()]
    return df


"""
# 并行计算会报错: no tables found
def get_financial_data(index_code="000300", sleep=0.01):
    def get_stock_data(code):
        stock_data = ak.stock_financial_analysis_indicator(symbol=code)
        stock_data["code"] = code + ".SH" if code[0] == "6" else code + ".SZ"
        stock_data.set_index(["日期", "code"], inplace=True)
        time.sleep(sleep)
        return stock_data

    cons = ak.index_stock_cons(symbol=index_code)
    df_list = Parallel(n_jobs=-1)(delayed(get_stock_data)(code) for code in cons["品种代码"])
    df = pd.concat(df_list, axis=0)
    df.dropna(axis=1, how="all", inplace=True)
    df.index.names = ["datetime", "code"]
    df = df[~df.index.duplicated()]
    return df
"""


def get_financial_data(index_code="000300"):
    df = pd.DataFrame()
    cons = ak.index_stock_cons(symbol=index_code)
    for code in cons["品种代码"]:
        stock_data = ak.stock_financial_analysis_indicator(symbol=code)
        stock_data["code"] = code + ".SH" if code[0] == "6" else code + ".SZ"
        df = pd.concat([df, stock_data], axis=0)
    df = df.set_index(["日期", "code"]).sort_index()
    df.dropna(axis=1, how="all", inplace=True)
    df.index.names = ["datetime", "code"]
    df = df[~df.index.duplicated()]
    return df


def get_futures_news(instrument="AL"):
    """
    由于期货是T0, 而新闻的datetime无法具体到分钟，而且新闻具有发布时间离散, 发布时集中(指同一天有多条新闻)的特点, 因此很难直接整合进行情数据中

    :param instrument: 品种代码, 由于akshare采用的方法是代码后面+888(表示指数合约), 因此只要输入合约代码的前两位即可
    :return: pd.DataFrame, 包括作为索引的datetime, instrument, 作为正式内容的新闻标题(akshare不返回正文内容)和正文链接

    注: 链接点开会404, 所以没什么用

    instrument 示例:
    AL: 沪铝
    J9: 焦炭
    TA: PTA
    CJ: 红枣
    JM: 焦煤
    """
    news = ak.futures_news_baidu(symbol=instrument)
    news.columns = ["title", "datetime", "link"]
    news["instrument"] = instrument
    return news.set_index(["datetime", "instrument"]).sort_index()


def get_high_freq_futures(instrument="PTA", freq=1):
    """
    :param instrument: 资产名称, 品种大类的中文名, 例如PTA, 白糖等
    :param freq: int, 频率, 1为1分钟, 以此类推
    :return: pd.DataFrame
    """
    all_contracts = ak.futures_zh_realtime(symbol=instrument)["symbol"].tolist()
    all_data = pd.DataFrame()
    for contract in all_contracts:
        data = ak.futures_zh_minute_sina(symbol=contract, period=str(freq))
        data["instrument"] = contract
        all_data = pd.concat([all_data, data], axis=0)
    all_data.dropna(axis=1, how="all", inplace=True)
    all_data.set_index(["datetime", "instrument"], inplace=True)
    return all_data.sort_index()
