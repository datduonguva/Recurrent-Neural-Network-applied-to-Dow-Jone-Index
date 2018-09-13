#Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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

def softmax(list_X):
    """Safely calculate softmax"""

    max_value = np.max(list_X)
    exp_list = np.exp(list_X - max_value)
    return exp_list/np.sum(exp_list)

def forward(x, params):
    U, V, W, T = params['U'], params['V'], params['W'], params['n_timestep']

    n_hidden = params['n_hidden']
    n_words = params['n_words']

    s = np.zeros((T+1, n_hidden))
    o = np.zeros((T, n_words))

    for t in range(T):
        s[t] = np.tanh(U[:,x[t]]+ W.dot(s[t-1]))
        o[t] = softmax(V.dot(s[t]))
    return o, s

def predict(x, params):
    """ x is a list of input"""
    o, s = forward(x, params)
    return np.argmax(o, axis = 1)

def loss(x, y, params):
    """ x is list of input, y is list of output """
    L = 0.0
    for i in range(len(y)):
        o, s = forward(x[i], params)
        correct_words = o[np.arange(len(y[i])), y[i]]
        L += -np.sum(np.log(correct_words))
    L = L/(len(y)*len(y[i]))
    return L

def bptt(x, y, params, steps=3):
    """ x, y is list or input and output WORDS"""
    T = len(y)
    o, s = forward(x, params)
    dLdU = np.zeros(params['U'].shape)
    dLdV = np.zeros(params['V'].shape)
    dLdW = np.zeros(params['W'].shape)
    delta_o = o
    delta_o[np.arange(T), y] -= 1 #<====== This may be wrong
    for t in range(T-1, -1, -1):
        dLdV += np.outer(delta_o[t], s[t])
        delta_t = params['V'].T.dot(delta_o[t])*(1-np.power(s[t], 2))
        for i in range(max(0, t-steps), t+1)[::-1]:
            dLdW += np.outer(delta_t, s[i -1])
            dLdU[:, x[i]] += delta_t
            delta_t = params['W'].T.dot(delta_t)*(1-np.power(s[i-1],2))
    return dLdU, dLdV, dLdW

def train(X_train, Y_train, X_val, Y_val,mparams,  learning_rate=0.01, epoch=20):
    """ Self-explained"""
    loss_history = []
    loss_val_history = []
    for e in range(epoch):
        loss_history.append(loss(X_train, Y_train, params))
        loss_val_history.append(loss(X_val, Y_val, params))
        if e > 5:
            if loss_val_history[e] > loss_val_history[e-2]:
                break
        print("epoch, train_loss, val_loss: {}, {}, {}".format(e, loss_history[-1], loss_val_history[-1]))
        for i in range(len(X_train)-1):
            dLdU, dLdV, dLdW = bptt(X_train[i], Y_train[i], params)
            params['U'] -= dLdU*learning_rate
            params['V'] -= dLdV*learning_rate
            params['W'] -= dLdW*learning_rate

if __name__ == '__main__':
    df = import_data()[['Date', 'Change']]
    df, n_words = encode_data(df)
    
    n_timestep = 5
    n_hidden = 100
    X, Y = create_training_data(df, n=n_timestep)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    
    #initialize hidden state:
    params = {'n_hidden': n_hidden, 'n_timestep': n_timestep, 'n_words': n_words}
    params['U'] = (2*np.random.rand(n_hidden, n_words)-1)/np.sqrt(n_words)
    params['V'] = (2*np.random.rand(n_words, n_hidden)-1)/np.sqrt(n_hidden)
    params['W'] = (2*np.random.rand(n_hidden, n_hidden) -1)/np.sqrt(n_hidden)
    if 'U.npy' not in os.listdir():
        train(X_train, Y_train, X_val, Y_val, params,  learning_rate=0.005, epoch=300)
        np.save('U.npy', params['U'])
        np.save('V.npy', params['V'])
        np.save('W.npy', params['W'])

    params['U'] = np.load('U.npy')
    params['V'] = np.load('V.npy')
    params['W'] = np.load('W.npy')
    for i in range(200):
        print(X[i], predict(X[i], params))