import pandas as pd
import tushare as ts
import os


def get_adj_hfq(price: pd.Series, pre_close: pd.Series) -> pd.Series:
    """
    计算后复权因子
    """
    price_ratio = (price / pre_close).groupby(level=1).transform(lambda x: x.cumprod())
    adj = price_ratio.groupby(level=1).transform(lambda x: x / x[0])
    return adj


def tus_init(tus_token: str = ""):
    token = tus_token
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


def get_index_cons(pro, index_code: str = "000905.SH", start: str = "20100101", end: str = "20101231",
                   output_folder: str = ""):
    data = pd.DataFrame()
    data.index.names = ['datetime']
    df = pd.DataFrame(pro.index_weight(index_code=index_code, start_date=start, end_date=end))  # 获得成分股列表
    df.set_index(['trade_date'], inplace=True)
    df.index.names = ['datetime']
    df = df.sort_index()
    data = pd.concat([data, df], axis=0).sort_index()
    data.to_csv(output_folder + 'index_weight.csv')


def process_index_cons(folder_path):
    files = os.listdir(folder_path)
    idx_cons = pd.DataFrame()

    for file in files:
        filepath = folder_path + file
        sub_df = pd.read_csv(filepath)
        sub_df.set_index("datetime", inplace=True)
        code_list = pd.DataFrame()
        codes = sub_df["con_code"].groupby(level=0).apply(lambda x: ','.join(x.astype(str)))
        code_list["ts_code"] = codes
        code_list["days"] = code_list.index.get_level_values(0)
        code_list["days"] = code_list["days"].astype(str)
        code_list["days"] = pd.to_datetime(code_list["days"], format="%Y-%m-%d")
        # print(code_list)
        code_list.reset_index(inplace=True)
        code_list.set_index("days", inplace=True)
        new_index = pd.date_range(start=code_list.index.min(), end=code_list.index.max(), freq='D')
        code_list = code_list.reindex(new_index)
        idx_cons = pd.concat([idx_cons, code_list], axis=0)
    idx_cons.sort_index(inplace=True)
    idx_cons.index.name = "days"
    idx_cons["datetime"] = idx_cons.index.get_level_values(0).strftime("%Y%m%d").astype(int)
    idx_cons.fillna(method="ffill", inplace=True)
    idx_cons.to_csv("instrument_list.csv")


def get_stock_data(pro, file_path='instrument_list.csv', adjust_price: bool = False) -> pd.DataFrame:
    instrument_data = pd.DataFrame()
    # 读取code_list后，按照list获取每支股票的数据
    df1 = pd.read_csv(file_path)
    df1.fillna(method='ffill', inplace=True)

    date = df1['datetime'].unique()
    day = []
    for i in range(len(date)):
        day.append(str(date[i]))

    for i in range(len(date)):
        df = pd.DataFrame(pro.daily(ts_code=str(df1['ts_code'].values[i]), start_date=day[i], end_date=day[i]))  # 行情数据
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index(['trade_date'], inplace=True)
        df.index.names = ['datetime']
        df = df.sort_index()
        instrument_data = pd.concat([instrument_data, df], axis=0).sort_index()
    instrument_data = instrument_data.reset_index()
    instrument_data.set_index(["datetime", "ts_code"], inplace=True)
    instrument_data.index.names = ["datetime", "instrument"]
    if adjust_price:
        adj = get_adj_hfq(instrument_data["close"], instrument_data["pre_close"])
        # fixme: 增加调整volume的功能
        prices = ["open", "close", "high", "low"]
        for p in prices:
            instrument_data[p] *= adj
    return instrument_data
