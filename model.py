#Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
def import_data_adding_off_days():
    """ Read the data,
        adding weekend date
    """
    df = None
    if 'DJI_processed.csv' in os.listdir():
        df = pd.read_csv('DJI_processed_no_off.csv', converters = {'Date':lambda x: datetime.strptime(x, '%Y-%m-%d')})
    else:
        df = pd.read_csv('DJI.csv')
        df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        delta = timedelta(1)
        dates = df['Date'].tolist()
        old_dates = df['Date'].tolist()
        cur_pos = 1
        while cur_pos <len(dates) :
            if dates[cur_pos] - dates[cur_pos-1] != delta:
                dates.insert(cur_pos, dates[cur_pos-1]+ delta)
            cur_pos += 1
        for i in range(len(dates)):
            if dates[i] not in old_dates: 
                tmp = df[df['Date'] == dates[i]-delta].copy()
                tmp['Date'] = dates[i]
                if len(tmp)!= 0:
                    df = df.append(tmp.iloc[0].copy())
        df = df.sort_values('Date')
    
        close_next =[0]+  df.iloc[0:-1]['Close'].tolist() 
        df['Close_Next'] = close_next
        df['Change'] = (df['Close_Next'] - df['Close'])/df['Close']
        df = df.iloc[1:]
        df.to_csv('DJI_processed_no_off.csv', index=False)
    return df

def import_data():
    """ Read the data,
        adding weekend date
    """
    df = None
    if 'DJI_processed.csv' in os.listdir():
        df = pd.read_csv('DJI_processed.csv', converters = {'Date':lambda x: datetime.strptime(x, '%Y-%m-%d')})
    else:
        df = pd.read_csv('DJI.csv')
        df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        df['Close_Next'] = [0]+  df.iloc[0:-1]['Close'].tolist() 
        df['Change'] = (df['Close_Next'] - df['Close'])/df['Close']
        df = df.iloc[1:]
        df.to_csv('DJI_processed.csv', index=False)
    return df


def encode_data(df):
    """
        adding to the dataframe
    """
    min_val = df['Change'].min()- 0.05
    max_val = df['Change'].max()+ 0.05
    resolution = 0.0001
    n = int((max_val-min_val)/resolution)
    df['Change_id'] = ((df['Change']-min_val)/resolution).astype(int)
    return df, n
def create_training_data(df, n=5):
    change_id_list = df['Change_id'].tolist()
    input_list = np.array([change_id_list[i:i+n] for i in range(len(change_id_list)-n-1)])
    output_list = np.array([change_id_list[i:i+n] for i in range(1, len(change_id_list)-n)])
    return input_list, output_list
    
    

if __name__ == '__main__':
    df = import_data()[['Date', 'Change']]
    df, n = encode_data(df)
    input_list, output_list = create_training_data(df)
    for i in range(10):
        print(input_list[i], output_list[i])

