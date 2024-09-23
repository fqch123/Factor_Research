import akshare as ak
import polars as pl
import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime
import jqdatasdk as jq
from Data_Fetcher import *

def fundamental_factors_calculation(stock_code:str,start_date:str,end_date:str):
    '''
    基本面因子计算
    '''
    # 获取2016-01-01至2023-12-31每月末的交易日列表
    trade_dates = GetTradePeriod(start_date, end_date)

    fundamental_factors = []
    # 遍历每个交易日,获取中证500指数成分股
    for date in trade_dates:
        stock_pool = get_stockpool(stock_code,date.strftime('%Y-%m-%d'))
        # 获取因子数据
        q = jq.query(
            jq.valuation.code,
            jq.valuation.market_cap,  # 总市值，用于规模因子
            jq.valuation.pb_ratio,    # 市净率，用于价值因子
            jq.indicator.roe         # 净资产收益率，用于盈利因子
        ).filter(jq.valuation.code.in_(stock_pool))
        df = jq.get_fundamentals(q, date=date) 
        df['date'] = date
        fundamental_factors.append(df)
    # 合并所有日期的因子数据
    fundamental_factors_df = pd.concat(fundamental_factors, ignore_index=True)
    # 重命名列
    fundamental_factors_df = fundamental_factors_df.rename(columns={
        'market_cap': 'size_factor',
        'pb_ratio': 'value_factor',
        'roe': 'profitability_factor'
    })
    #计算BM值
    fundamental_factors_df['value_factor'] = 1./fundamental_factors_df['value_factor']
    return fundamental_factors_df



def price_factors_calculation(df:pl.DataFrame):
    '''
    量价因子计算
    '''
    df = df.with_columns([
        (pl.col("收盘").pct_change().over("股票代码").rolling_mean(10)).alias("10日动量"),
        (pl.col("收盘").pct_change().over("股票代码").rolling_std(20)).alias("20日波动率"),
        (pl.col("成交量").log() - pl.col("成交量").over("股票代码").rolling_mean(20).log()).alias("相对成交量"),
        (pl.col("收盘") / pl.col("收盘").over("股票代码").rolling_mean(60) - 1).alias("60日相对强弱"),
        ((pl.col("收盘").pct_change().over("股票代码").abs())/ (pl.col("成交量").pct_change().over("股票代码").abs() + 1e-8)).alias("价量比"),
    ])
    return df


def rolling_correlation(x, y, window):
    '''
    辅助函数，计算滚动相关系数
    '''
    x_mean = x.rolling_mean(window)
    y_mean = y.rolling_mean(window)
    numerator = ((x - x_mean) * (y - y_mean)).rolling_sum(window)
    denominator = (((x - x_mean)**2).rolling_sum(window) * ((y - y_mean)**2).rolling_sum(window)).sqrt()
    return numerator / denominator

def alpha_002(stock_data):
    # 按日期分组，对每个日期进行操作
    return (
        stock_data
        .sort(['日期', '股票代码'])
        .groupby('日期')
        .apply(lambda group: 
            pl.DataFrame({
                '日期': group['日期'],
                '股票代码': group['股票代码'],
                'volume_rank': group['成交量'].cast(pl.Float64).log().diff(2).rank(),
                'price_rank': ((group['收盘'] - group['开盘']) / group['开盘']).rank()
            })
        )
        .sort(['股票代码', '日期'])
        .groupby('股票代码')
        .apply(lambda group: 
            pl.DataFrame({
                '日期': group['日期'],
                '股票代码': group['股票代码'],
                'alpha002': -1 * rolling_correlation(group['volume_rank'], group['price_rank'], window=6)
            })
        )
        .sort(['股票代码', '日期'])
    )


def alpha_003(stock_data):
    return (
        stock_data
        .sort(['日期', '股票代码'])
        .groupby('日期')
        .apply(lambda group: 
            pl.DataFrame({
                '日期': group['日期'],
                '股票代码': group['股票代码'],
                'open_rank': group['开盘'].rank(),
                'volume_rank': group['成交量'].rank()
            })
        )
        .sort(['股票代码', '日期'])
        .groupby('股票代码')
        .apply(lambda group: 
            pl.DataFrame({
                '日期': group['日期'],
                '股票代码': group['股票代码'],
                'alpha003': -1 * rolling_correlation(group['open_rank'], group['volume_rank'], window=10)
            })
        )
        .sort(['股票代码', '日期'])
    )

if __name__ == '__main__':
    jq.auth('user', 'password')  # 请替换为自己的聚宽账号和密码
    fundamental_factors_df = fundamental_factors_calculation('000905.XSHG','2016-01-01','2023-12-31')
    output_file = '基本面因子.xlsx'
    fundamental_factors_df.to_excel(output_file,index=False)
    

