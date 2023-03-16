import pandas as pd
import json
import requests
import datetime
from datetime import timedelta

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