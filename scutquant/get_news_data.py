import pandas as pd
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from joblib import Parallel, delayed

"""
此模块适用于获取那些结构为 base_url + day + page 的报纸新闻
This module is designed to retrieve newspaper news with a url structure of "base_url + day + page".

A simple example:

BaseURL = "http://epaper.cenews.com.cn/html/"  # 中国环境报, 有可能被封ip
page = "node_2.htm"

result = parallel_pipeline(BaseURL, sample_page=page, start="2023-01-01", end="2023-05-29",
                           page_selector="tr > td > a#pageLink", title_selector='tr > td:nth-of-type(2).px12 > a',
                           article_selector="div#ozoom", n_jobs=-1)
                           
result.to_csv("news_env.csv", encoding="utf-8-sig)
"""


def get_calendar(start, end, strftime="%Y-%m-%d"):
    """
    给定起始日期和结束日期, 生成中间的所有日期, 按strftime格式输出
    :param start: str
    :param end: str
    :param strftime:
    :return:
    """
    start = datetime.strptime(start, strftime)
    end = datetime.strptime(end, strftime)
    delta = timedelta(days=1)
    date = start
    calendar = []
    while date <= end:
        calendar.append(date.strftime(strftime))
        date += delta
    return calendar


def get_total_pages(sample_url, encoding="utf-8", select='div.swiper-container > div.swiper-slide > a', href="href"):
    """
    场景: 中国能源报电子版的每一期分n版, 点开随便一版都能看到其它版的url, 现在需要获取每一期所有版的信息, 即要获取所有版的url
    :param sample_url:
    :param encoding:
    :param select: 目标值的标签位置
    :param href: 获取的目标, 例如"href"或者“data-href”
    :return:
    """
    response = requests.get(sample_url)
    if response.status_code != 200:
        # print("error occurs")
        return []
    else:
        response.encoding = encoding
        soup = BeautifulSoup(response.text, "html.parser")
        news_list = soup.select(select)
        total_pages = [news[href] for news in news_list]
        # total_pages = total_pages.append(sample_url) if sample_url not in total_pages else total_pages
        # print(total_pages)
        return total_pages


def get_title(url, encoding="utf-8", select="div.news > ul > li > a", href="href"):
    """
    获取报纸每一期每一版下的所有标题
    :param url:
    :param encoding:
    :param select: 目标值的标签位置
    :param href: 新闻文本的url元素名
    :return:
    """
    response = requests.get(url)
    if response.status_code != 200:
        return []
    else:
        response.encoding = encoding
        soup = BeautifulSoup(response.text, "html.parser")
        news_list = soup.select(select)
        news, article_href = [news.text for news in news_list], [news[href] for news in news_list]
        return news, article_href


def get_article(url, encoding="utf-8", select="div#articleContent"):
    response = requests.get(url)
    if response.status_code != 200:
        return []
    else:
        response.encoding = encoding
        soup = BeautifulSoup(response.text, "html.parser")
        news_list = soup.select(select)
        return [article.text for article in news_list]


def process_datetime_rmrb(day, process=True):
    """
    将日期改造成人民日报网页的形式(%Y-%m/%d)
    :param day:
    :param process: 是否整理成人民日报系列的格式, 如果否则返回原格式
    :return:
    """
    if process:
        day_in_format = day[:7] + "/" + day[8:]
    else:
        day_in_format = day
    return day_in_format


def process_url(base: str, day: str, tail: str, symbol="/", whb=False) -> str:
    """
    一个完整的url分成3部分: 开头(base), 日期(day), 还有结尾(tail). day和tail用symbol隔开
    :param whb: 如果是爬文汇报, 则需要另外的处理如下
    :param base:
    :param day:
    :param tail:
    :param symbol:
    :return:
    """
    if whb:
        return base + tail
    else:
        return base + day + symbol + tail


def dic2df(dic):
    """
    将一个字典:
    {datetime: {title: article}}转成具有datetime, title和article三列的pd.DataFrame
    :param dic:
    :return:
    """
    df_list = []
    for dt, titles in dic.items():
        for title, articles in titles.items():
            for article in articles:
                df_list.append({'datetime': dt, 'title': title, 'article': article})
    df = pd.DataFrame(df_list)
    return df


# parallel_pipeline的原型, 用于实验
def pipeline(base_url, sample_page, start, end, strftime="%Y-%m-%d", page_selector="ul > li > a#pageLink",
             title_selector="div#titleList > ul > li > a", article_selector="div#articleContent", symbol="/",
             href_page="href", href_article="href", process_day=True, whb=False):
    """
    :param base_url: eg:http://epaper.cenews.com.cn/html/, 一家报纸的url的开头部分
    :param sample_page: eg:node_2.htm, 一家报纸的url的结尾部分
    :param start: 开始日期
    :param end: 结束日期
    :param strftime: 日期格式
    :param page_selector: page所在的位置
    :param title_selector: title所在的位置
    :param article_selector: 新闻正文所在的位置
    :param symbol: day和tail之间的间隔符
    :param href_page: 获取的page的href
    :param href_article: 获取正文的href
    :param process_day: 是否将日期处理成人民日报的格式
    :param whb: 如果是文汇报, 需要特别的处理
    :return: pd.DataFrame
    """
    calendar = get_calendar(start, end, strftime)
    all_results = {}
    for day in calendar:
        all_results[day] = {}
        day_in_format = process_datetime_rmrb(day=day, process=process_day)
        s_url = process_url(base_url, day_in_format, sample_page, symbol=symbol)
        # print(s_url)
        pages = get_total_pages(s_url, select=page_selector, href=href_page)
        # print(pages)
        if len(pages) > 0:  # 如果是空列表则跳过
            if not whb:
                pages[0] = pages[0][2:]
            for p in pages:
                target_url = process_url(base_url, day_in_format, p, symbol=symbol, whb=whb)
                print(target_url)
                title, article_href = get_title(target_url, select=title_selector, href=href_article)
                for a_id in range(len(article_href)):
                    a_url = process_url(base_url, day_in_format, article_href[a_id], symbol=symbol, whb=whb)
                    print(a_url)
                    article = get_article(a_url, select=article_selector)
                    # print(article)
                    all_results[day][title[a_id]] = article
    all_results_df = dic2df(all_results)
    return all_results_df


def parallel_pipeline(base_url, sample_page, start, end, strftime="%Y-%m-%d", page_selector="ul > li > a#pageLink",
                      title_selector="div#titleList > ul > li > a", article_selector="div#articleContent", n_jobs=-1,
                      process_day=True, symbol="/", encoding="utf-8", href_page="href", href_article="href", whb=False):
    """
    流程如下:
    1、获取从start到end的所有日期
    2、对于每一天, 根据sample_url获取当期报纸的所有版的url
    3、对于每个url, 获取所有标题
    4、对于每个标题, 获取所有正文内容
    5、整理成pd.DataFrame格式并输出

    :param base_url: url的前半段, 一般是声明是哪家报纸
    :param sample_page: url的最后部分, 需要获取一个完整的url, 才能知道当期报纸有多少版
    :param start: 开始日期
    :param end: 结束日期
    :param strftime: 日期格式
    :param page_selector: page在页面中的位置
    :param title_selector: 标题在页面中的位置
    :param article_selector: 文章在页面中的位置
    :param n_jobs: 同时调用的cpu数
    :param process_day: 网页url的日期格式可能不是strftime支持的格式，故需要额外调整
    :param symbol: base, day和page中间的分隔符
    :param encoding: 编码格式, 一般为utf-8, 也可尝试utf-8-sig
    :param href_page: 获取的page的href
    :param href_article: 获取正文的href
    :param whb: 如果是文汇报则需要特别的处理
    """
    calendar = get_calendar(start, end, strftime)

    def process(day):
        all_results = {}
        all_results[day] = {}
        day_in_format = process_datetime_rmrb(day, process=process_day)
        s_url = process_url(base_url, day_in_format, sample_page, symbol=symbol)
        pages = get_total_pages(s_url, select=page_selector, encoding=encoding, href=href_page)
        if len(pages) > 0:  # 如果是空列表则跳过
            if not whb:
                pages[0] = pages[0][2:]
            for p in pages:
                target_url = process_url(base_url, day_in_format, p, symbol=symbol, whb=whb)
                title, article_href = get_title(target_url, select=title_selector, encoding=encoding, href=href_article)
                for a_id in range(len(article_href)):
                    a_url = process_url(base_url, day_in_format, article_href[a_id], symbol=symbol, whb=whb)
                    article = get_article(a_url, select=article_selector, encoding=encoding)
                    all_results[day][title[a_id]] = article
        return all_results

    results = Parallel(n_jobs=n_jobs)(delayed(process)(day) for day in calendar)
    all_results_dict = {}
    for d in results:
        all_results_dict.update(d)

    all_results_df = dic2df(all_results_dict)
    return all_results_df


def merge_news_data(folder_path, target_dir, name, encoding="utf-8-sig", index="datetime"):
    """
    example:

    merge_news_data("D:/Desktop/whb/", "D:/Desktop/news/", name="whb")

    :param folder_path: str, 文件夹地址
    :param target_dir: str, 目标地址
    :param name: str, 文件名
    :param encoding:
    :param index:
    :return:
    """
    files = os.listdir(folder_path)

    df = pd.DataFrame()
    for file in files:
        target_dir = folder_path + file
        data = pd.read_csv(target_dir, encoding=encoding)
        data.set_index(index, inplace=True)
        df = pd.concat([df, data], axis=0)

    df = df.sort_index()
    df = df.iloc[:, 1:]
    print(df.head(5))
    df.to_csv(target_dir + name + ".csv", encoding=encoding)


def get_whb_data(start: str, end: str) -> pd.DataFrame:
    """
    获取文汇报的新闻

    :param start: %Y-%m-%d
    :param end: %Y-%m-%d
    :return: 包括三列: datetime, title, article
    """
    result = parallel_pipeline(base_url="http://dzb.whb.cn/", sample_page="1/index.html#1685452485529", start=start,
                               end=end, page_selector="a.TitleInfo", title_selector="div.title_box.list > a",
                               article_selector="div#newsContent.news_content", href_page="data-href",
                               process_day=False, whb=True)
    return result


def get_hjb_data(start: str, end: str) -> pd.DataFrame:
    """
    获取中国环境报的新闻(可能会封ip!)

    :param start: %Y-%m-%d
    :param end: %Y-%m-%d
    :return: 包括三列: datetime, title, article
    """
    result = parallel_pipeline("http://epaper.cenews.com.cn/html/", sample_page="node_2.htm", start=start, end=end,
                               page_selector="tr > td > a#pageLink", title_selector='tr > td:nth-of-type(2).px12 > a',
                               article_selector="div#ozoom", n_jobs=-1, encoding="utf-8-sig")
    return result


def get_gmrb_data(start: str, end: str) -> pd.DataFrame:
    """
    获取光明日报的新闻

    :param start: %Y-%m-%d
    :param end: %Y-%m-%d
    :return: 包括三列: datetime, title, article
    """
    result = parallel_pipeline("https://epaper.gmw.cn/gmrb/html/", sample_page="nbs.D110000gmrb_01.htm", start=start,
                               end=end)
    return result


def get_zqb_data(start: str, end: str) -> pd.DataFrame:
    """
    获取中国青年报的新闻

    :param start: %Y-%m-%d
    :param end: %Y-%m-%d
    :return: 包括三列: datetime, title, article
    """
    result = pipeline("http://zqb.cyol.com/html/", sample_page="nbs.D110000zgqnb_01.htm", start=start, end=end)
    return result


def get_rmrb_data(start: str, end: str) -> pd.DataFrame:
    """
    获取人民日报的新闻

    :param start: %Y-%m-%d
    :param end: %Y-%m-%d
    :return: 包括三列: datetime, title, article
    """
    result = parallel_pipeline("http://paper.people.com.cn/rmrb/html/", sample_page="nbs.D110000renmrb_01.htm",
                               page_selector="a#pageLink", title_selector="ul >li > a", start=start, end=end)
    return result


def get_zjrb_data(start: str, end: str) -> pd.DataFrame:
    """
    获取浙江日报的新闻

    :param start: %Y-%m-%d
    :param end: %Y-%m-%d
    :return: 包括三列: datetime, title, article
    """
    result = parallel_pipeline("http://zjrb.zjol.com.cn/html/", sample_page="node_18.htm",
                               page_selector="a#pageLink", title_selector="ul.main-ed-articlenav-list > li > a",
                               article_selector="div#ozoom > founder-content", start=start, end=end)
    return result
