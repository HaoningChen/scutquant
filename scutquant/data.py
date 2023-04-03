import akshare as ak
import pandas as pd
import datetime


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
    return df.set_index(["日期", "code"]).sort_index()


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

