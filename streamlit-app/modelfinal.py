import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from datetime import timedelta, date
import scipy.optimize as sco
import parse 

data1 = parse.loader_and_cleaner('GLTR')
tickers = ['RTKM', 'MDMG', 'OZON', 'SBER', 'RTKMP', 'TATN' , 'ROSN' , 'LKOH' , 'PLZL' , 'MOEX', 'YNDX', 'MGNT' , 'VTBR' ,  'MTSS', 'FEES' , 'TRNFP' , 'RUAL' , 'AFLT']
for tic in tickers:
    data =  parse.loader_and_cleaner(f"{tic}")
    data1 = data1.merge(data, on=["TRADEDATE"])

stocks = ['GLTR','RTKM', 'MDMG', 'OZON', 'SBER', 'RTKMP', 'TATN' , 'ROSN' , 'LKOH' , 'PLZL' , 'MOEX', 'YNDX', 'MGNT' , 'VTBR' ,  'MTSS', 'FEES' , 'TRNFP' , 'RUAL' , 'AFLT']

data1 = data1.set_index('TRADEDATE')

def plot():
    (data1 / data1.iloc[0] * 100).plot(figsize=(15, 10))
    return plt.show()

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


print("Распределение долей акций в портфеле с максимальным коэффициентом Шарпа:\n")
print("Годовая доходность:", round(rp,10))
print("Годовой риск:", round(sdp,3))
print("Коэффициент Шарпа:", round((rp - risk_free_rate)/sdp, 3))
print(max_sharpe_allocation)


print("Распределение долей акций в портфеле с наименьшим показателем риска:\n")
print("Годовая доходность:", round(rp_min,10))
print("Годовой риск:", round(sdp_min,3))
print("Коэффициент Шарпа:", round((rp_min - risk_free_rate)/sdp_min, 3))
print(min_vol_allocation)