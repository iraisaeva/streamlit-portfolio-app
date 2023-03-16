import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import json
import requests
import pandas as pd
import numpy as np
from datetime import timedelta
import scipy.optimize as sco
import datetime

with open('streamlit-app\my_app\style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


image = Image.open('streamlit-app\my_app\pages\logo.png')

st.image(image)

st.text("")
def loader_and_cleaner(ticker):

    base_url = "http://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities.json"
    response = requests.get(base_url)
    result = json.loads(response.text)
    col_name = result['history']['columns']
    data_shares = pd.DataFrame(columns = col_name)

    url_share = f'http://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json'
    
    response = requests.get(url_share)
    result = json.loads(response.text) 
    resp_date = result['history']['data'] 
   
    data_shares  = pd.DataFrame(resp_date, columns = col_name)
    
    a = len(resp_date)

    b = 100
    while a == 100:
        url_opt = '?start=' + str(b)
        url_next_page  = url_share + url_opt
        response = requests.get(url_next_page)
        result = json.loads(response.text)
        resp_date = result['history']['data']
        data_next_page = pd.DataFrame(resp_date, columns = col_name)
        data_shares = pd.concat([data_shares, data_next_page], ignore_index=True) 
        a = len(resp_date)
        b = b + 100

    data_shares['TRADEDATE'] = pd.to_datetime(data_shares['TRADEDATE'])
    days=365
    today = datetime.date.today()
    dt_from = today - timedelta(days=days)
    today = pd.to_datetime(today)
    dt_from = pd.to_datetime(dt_from)

    data =  data_shares.loc[(data_shares['TRADEDATE'] >= dt_from) & (data_shares['TRADEDATE'] < today)]

    data = data[['TRADEDATE','CLOSE']]
    data = data.rename(columns={"CLOSE": f"{ticker}"})
    
    data.dropna(inplace = True)
    
    
    return data


data_load_state = st.text('Загружаем информацию...')


data1 = loader_and_cleaner('GLTR')
tickers = ['RTKM', 'MDMG', 'OZON', 'SBER', 'RTKMP', 'TATN' , 'ROSN' , 'LKOH' , 'PLZL' , 'MOEX', 'YNDX', 'MGNT' , 'VTBR' ,  'MTSS', 'FEES' , 'TRNFP' , 'RUAL' , 'AFLT']
for tic in tickers:
    data =  loader_and_cleaner(f"{tic}")
    data1 = data1.merge(data, on=["TRADEDATE"])


stocks = ['GLTR','RTKM', 'MDMG', 'OZON', 'SBER', 'RTKMP', 'TATN' , 'ROSN' , 'LKOH' , 'PLZL' , 'MOEX', 'YNDX', 'MGNT' , 'VTBR' ,  'MTSS', 'FEES' , 'TRNFP' , 'RUAL' , 'AFLT']


data1 = data1.set_index('TRADEDATE')

st.line_chart(data1 / data1.iloc[0] * 100)

data_load_state.text('Информация... Загружена!')

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
num_portfolios = 10000 
risk_free_rate = 0.00 
num_periods_annually = 1 
returns = data1.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) * num_periods_annually
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(num_periods_annually)
    return std, returns


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
min_vol = min_variance(mean_returns, cov_matrix)
sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
max_sharpe_allocation = pd.DataFrame(max_sharpe.x.copy(),index=data1.columns,columns=['allocation'])
max_sharpe_allocation.allocation = [round(i*100,2) for i in max_sharpe_allocation.allocation]
max_sharpe_allocation = max_sharpe_allocation.T
sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
min_vol_allocation = pd.DataFrame(min_vol.x.copy(),index=data1.columns,columns=['allocation'])
min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
min_vol_allocation = min_vol_allocation.T
target = np.linspace(rp_min, 0.00081, 20)
efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)


results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
choise = st.selectbox(
    'Выберите стратегию сбора портфеля:', ('минимальный риск', 'максимальный доход'))
data_load_state = st.text('Загружаем информацию...')
if choise == 'максимальный доход':
    r1=round(rp* 100,4)
    r2=round(sdp* 100,3)
    r3=round((rp - risk_free_rate)/sdp * 100, 3)

    st.text("Распределение долей акций в портфеле с максимальным доходом:\n")
    st.text(f"Годовая доходность: {r1} %")
    st.text(f"Годовой риск: {r2} % ")
    st.text(max_sharpe_allocation)
if choise == 'минимальный риск':
    r4 =round(rp_min * 100,4)
    r5= round(sdp_min * 100,3)
    r6 = round((rp_min - risk_free_rate)/sdp_min * 100, 3)
    st.text("Распределение долей акций в портфеле с наименьшим показателем риска:\n")
    st.text(f"Годовая доходность: {r4} %")
    st.text(f"Годовой риск: {r5} %")
    st.text(min_vol_allocation)

data_load_state.text('Информация... Загружена!')
