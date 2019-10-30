

import requests
import pprint
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from patsy import dmatrices
from datetime import datetime as dt
from datetime import timedelta

def __convert_timeseries(column_pos, df, metric_name, prom_metric, labels):
    for value in prom_metric['values']:
        timestamp = dt.utcfromtimestamp(int(value[0]))
        value = float(value[1])
        for c in labels:
             df.at[column_pos, c] = labels[c]
        df.at[column_pos, 'time'] = timestamp
        df.at[column_pos, metric_name] = value
        column_pos += 1

def data_from_prom_api_response(prom_api_response):
    data_frame = pd.DataFrame()

    if "data" not in prom_api_response:
        return data_frame

    ts_raw_data = prom_api_response['data']['result']

    col_pos_index = 0
    anonymous_metric_counter = 0

    for prom_metric in ts_raw_data:
        labels = {}
        metric_name = ""
        for l in prom_metric['metric']:
            if l == '__name__':
                
                metric_name = prom_metric['metric'][l]
            else:
                labels[l] = prom_metric['metric'][l]
        __convert_timeseries(col_pos_index, data_frame, metric_name, prom_metric, labels)

    #if data_frame.size > 0:
    #    data_frame.set_index('time', inplace=True)
    #data_frame.sort_index(inplace=True)
    return data_frame

def getData(hostname, match, start, end, step=60):
    headers = {}
    params = {
        'query': match,
        'start': start,
        'end' : end,
        'step': step
    }

    res = requests.get('http://' + hostname + '/api/v1/query_range', params=params, headers=headers)
    data  = res.json()
    df = data_from_prom_api_response(data)
    pprint.pprint(df)
    modeltraining(df, 'up ~ job + instance + time')
    
    
def modeltraining(df, formula):
    y, X = dmatrices(formula, data=df, return_type='dataframe')
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    #Y_train = list(data.get('value'))
    #X_train = list(data.get('ts'))
    #Y_train = data['value']
    #X_train = data['ts']
    #X_train = sm.add_constant(X_train)
    #model = sm.OLS(endog=Y_train, exog=X_train)
    #results = model.fit()
    #pprint.pprint(results)

    #prediction of future values
    #future_value = range(int(end_time), int(end_time) + 10000, 60)
    #X_pred = pd.DataFrame(data=future_value, columns=['ts'])
    #X_pred = sm.add_constant(X_pred)
    #prediction = model.predict(results.params, X_pred)
    
    #plot
    #plt.figure()
    #plt.plot(X_train['value'], model.predict(results.params), '-r', label='Linear model')
    #plt.plot(X_pred['value'], prediction, '--r', lable='Linear prediction')
    #plt.scatter(df['value'], df['ts'], label='data')
    #plt.xlabel('ts time')
    #plt.legend
    #plt.show


def createDataFrame(prom_data, column_order=[]):
#    pprint.pprint(prom_data)
    df = pd.DataFrame(prom_data,columns=column_order)
    return df

def main():
    prometheus_hostname = 'localhost:9090'
    metric_name = 'up'
    start = '1570815000'
    end = '1571419800'
    getData(prometheus_hostname, metric_name, start, end)

if __name__ == '__main__':
    main()
