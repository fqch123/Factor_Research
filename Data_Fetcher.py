import akshare as ak
import polars as pl
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime
import jqdatasdk as jq

#获取交易日
def GetTradePeriod(start_date: str, end_date: str, freq: str = 'ME') -> list:
    '''
    start_date/end_date:str YYYY-MM-DD
    freq:D天,M月,Q季,Y年 默认ME E代表期末 S代表期初
    ================
    return  list[datetime.date]
    '''
    # 获取交易日期并转换为polars DataFrame
    trade_days = ak.tool_trade_date_hist_sina()
    days = pl.DataFrame({'date': [d.strftime('%Y-%m-%d') for d in trade_days['trade_date'].values]})
    # 将日期字符串转换为datetime对象
    days = days.with_columns(pl.col('date').str.strptime(pl.Datetime,'%Y-%m-%d'))
    start_date = datetime.strptime(start_date,'%Y-%m-%d')
    end_date = datetime.strptime(end_date,'%Y-%m-%d')
    # 过滤日期范围
    days = days.filter((pl.col('date') >= start_date) & (pl.col('date') <= end_date))
    if freq == 'D':
        days = days.with_columns(pl.col('date').dt.date())
        return days.sort('date')['date'].to_list()
    # 添加年月列用于分组
    days = days.with_columns([
        pl.col('date').dt.year().alias('year'),
        pl.col('date').dt.month().alias('month')
    ])

    # 根据频率进行重采样
    if freq.startswith('M'):
        grouped = days.groupby(['year', 'month'])
    elif freq.startswith('Q'):
        grouped = days.groupby(['year', (pl.col('month') - 1) // 3])
    elif freq.startswith('Y'):
        grouped = days.groupby('year')
    else:
        raise ValueError("不支持的频率")

    if freq.endswith('E'):
        day_range = grouped.agg(pl.col('date').max())
    else:
        day_range = grouped.agg(pl.col('date').min())

    # 将datetime对象转换回日期对象
    day_range = day_range.with_columns(pl.col('date').dt.date())

    # 返回日期列表
    return day_range.sort('date')['date'].to_list()



# 获取股票池
def get_stockpool(symbol: str, watch_date: str) -> list:
    '''获取股票池'''
    #'A'代表全市场
    if symbol == 'A':
        stockList = jq.get_index_stocks('000002.XSHG', date=watch_date) + jq.get_index_stocks(
            '399107.XSHE', date=watch_date)
    else:
        stockList = jq.get_index_stocks(symbol, date=watch_date)
    stockList = del_st_stock(stockList, watch_date)  # 过滤ST
    stockList = del_iponday(stockList, watch_date)  # 过滤上市不足60日
    stockList = del_paused(stockList, watch_date)  # 过滤过去20日内曾停牌股票
    return stockList

def del_st_stock(securities: list, watch_date: str) -> list:
    '''过滤ST股票'''

    filtered = jq.get_extras('is_st', securities,
                          end_date=watch_date, df=True, count=1).iloc[0]

    return filtered[filtered == False].dropna().index.tolist()

def del_iponday(securities: list, watch_date: str, N: int = 60) -> list:
    '''返回上市大于N日的股票'''
    return list(filter(lambda x: (parse(watch_date).date() - jq.get_security_info(x, date=watch_date).start_date).days > N, securities))


def del_paused(securities: list, watch_date: str, N: int = 21) -> list:
    '''返回N日内未停牌的股票'''
    pausd_df = jq.get_price(securities, end_date=watch_date,
                         count=N, fields='paused', panel=False)
    cond = pausd_df.groupby('code')['paused'].sum()

    return cond[cond == 0].dropna().index.tolist()


def load_csi500_components(file_path: str) -> pl.DataFrame:
    """加载中证500成分股数据"""
    return pl.read_excel(file_path)

def load_stock_data(stock_codes: list, start_date: str, end_date: str) -> pl.DataFrame:
    """使用akshare加载股票数据"""
    data = []
    for code in stock_codes:
        df = ak.stock_zh_a_hist(symbol=code[:6], start_date=start_date, end_date=end_date, adjust="hfq")
        df['股票代码'] = code
        df = pl.from_pandas(df)
        data.append(df)
    return pl.DataFrame(pl.concat(data))

def load_index_data(start_date: str, end_date: str) -> pl.DataFrame:
    """加载中证500指数数据"""
    index_data = ak.index_zh_a_hist(symbol="000905",start_date=start_date,end_date=end_date)
    return pl.DataFrame(index_data)


if __name__ == '__main__':
    jq.auth('user', 'password')  # 请替换为自己的聚宽账号和密码
    # 获取2016-01-01至2023-12-31每月末的交易日列表
    trade_dates = GetTradePeriod('2016-01-01', '2023-12-31')

    # 创建一个空的DataFrame来存储结果
    result_df = pd.DataFrame() 
    # 遍历每个交易日,获取中证500指数成分股
    for date in trade_dates:
        stock_pool = jq.get_stockpool('000905.XSHG',date.strftime('%Y-%m-%d'))
        temp_df = pd.DataFrame({'date': date, 'stock_code': stock_pool})
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    # 将结果保存为Excel文件
    output_file = '中证500指数成分股_月末_2016-2023.xlsx'
    result_df.to_excel(output_file, index=False)
