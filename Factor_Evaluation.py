import akshare as ak
import polars as pl
import numpy as np
import pandas as pd
import empyrical as ep
from dateutil.parser import parse
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.family'] = ['sans-serif']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('seaborn-v0_8')


def get_group(df:pd.DataFrame,target_factor:str,num_group:int=5)->pd.DataFrame:
    
    '''
    分组
    ----
        target_factor:目标列
        num_group:默认分5组
    '''
    df = df.copy()
    label = ['G%s' % i for i in range(1, num_group+1)]
    df['group'] = df.groupby(level='date')[target_factor].transform(
    lambda x: pd.qcut(x, num_group, labels=label,duplicates='drop'))
    
    return df


def get_algorithm_return(factor_df:pd.DataFrame)->pd.DataFrame:
    
    '''
    获取分组收益率
    '''
    returns = pd.pivot_table(factor_df.reset_index(
    ), index='date', columns='group', values='next_month_return')
    returns.columns = [str(i) for i in returns.columns]

    returns.index = pd.to_datetime(returns.index)
    
    return returns


def _adjust_returns(returns, adjustment_factor):
    #计算调整后的收益率
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns.copy()
    return returns - adjustment_factor

def information_ratio(returns, factor_returns):
    '''
    信息比率
    '''
    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


def Strategy_performance(return_df: pd.DataFrame, periods='monthly') -> pd.DataFrame:
    '''计算风险指标 默认为月度:月度调仓'''

    ser: pd.DataFrame = pd.DataFrame()
    ser['年化收益率'] = ep.annual_return(return_df, period=periods)
    ser['波动率'] = return_df.apply(lambda x: ep.annual_volatility(x,period=periods))
    ser['夏普'] = return_df.apply(ep.sharpe_ratio, period=periods)
    ser['最大回撤'] = return_df.apply(lambda x: ep.max_drawdown(x))
    
    if 'benchmark' in return_df.columns:

        select_col = [col for col in return_df.columns if col != 'benchmark']

        ser['IR'] = return_df[select_col].apply(
            lambda x: information_ratio(x, return_df['benchmark']))
        ser['Alpha'] = return_df[select_col].apply(
            lambda x: ep.alpha(x, return_df['benchmark'], period=periods))

    return ser.T

def plot_nav(nav: pd.DataFrame, title: str):

    '''
    绘制分组净值曲线
    '''
    plt.figure(figsize=(18, 6))
    # 设置标题
    plt.title(title)
    # 1,5组设置不同的颜色和线型方便查看单调性
    plt.plot(nav['G1'], color='Navy', label='G1')
    plt.plot(nav['G2'], color='LightGrey', ls='-.', label='G2')
    plt.plot(nav['G3'], color='DimGray', ls='-.', label='G3')
    plt.plot(nav['G4'], color='DarkKhaki', ls='-.', label='G4')
    plt.plot(nav['G5'], color='LightSteelBlue', label='G5')
        
    plt.axhline(1, color='black', lw=0.5)
    plt.plot(nav['excess_ret'], color='r', ls='--', label='excess_ret')
    
    plt.plot(nav['benchmark'],color='black',ls='--',label='benchmark')
    
    # 设置x轴和y轴标签
    plt.xlabel('日期')
    plt.ylabel('累积收益')
    
    # 图例
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()


def ic_plot(merged_factors:pd.DataFrame,factor:str):
    #绘制IC图
    al_factor = merged_factors.rename(
    columns={factor: 'factor', 'next_month_return': 'forward_return'}).loc[:,['factor','forward_return']]
    al_factor = al_factor.dropna() 
    al_factor.index = al_factor.index.set_levels(pd.to_datetime(al_factor.index.levels[0]),level=0)

    # 计算每日 IC
    daily_ic = al_factor.groupby(level='date').apply(lambda x: x['factor'].corr(x['forward_return']))

    # 计算累积 IC
    cumulative_ic = daily_ic.cumsum()

    # 计算 IC 的均值和标准差
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()


    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # 绘制 IC 时间序列图
    ax1.plot(daily_ic.index, daily_ic.values, linewidth=1, label='Month IC')
    ax1.axhline(y=ic_mean, color='r', linestyle='--', linewidth=1, label=f'Mean IC: {ic_mean:.4f}')
    ax1.fill_between(daily_ic.index, ic_mean - ic_std, ic_mean + ic_std, alpha=0.2, color='r', label=f'±1 Std Dev: {ic_std:.4f}')
    ax1.set_title('Information Coefficient (IC) Time Series', fontsize=16)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('IC', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 绘制累积 IC 图
    ax2.plot(cumulative_ic.index, cumulative_ic.values, linewidth=1)
    ax2.set_title('Cumulative Information Coefficient', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative IC', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.show()
