import akshare as ak
import pandas as pd
import datetime

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
    df = df[~df.index.duplicated()]
    data.columns = ["open", "close", "high", "low", "volume", "amount", "amplitude", "price_chg", "pct_chg", "turnover"]
    return data


def get_high_freq_data(index_code="000300", minutes=1, adjust="hfq"):
    cons = ak.index_stock_cons(symbol=index_code)
    df = pd.DataFrame()
    for code in cons["品种代码"]:
        stock_code = "sh" + code if code[0] == "6" else "sz" + code
        stock_data = ak.stock_zh_a_minute(symbol=stock_code, period=str(minutes), adjust=adjust)
        stock_data["code"] = stock_code
        df = pd.concat([df, stock_data], axis=0)
    df = df.set_index(["day", "code"]).sort_index()
    df = df[~df.index.duplicated()]
    return df
    