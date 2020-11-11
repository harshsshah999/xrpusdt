import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, LeakyReLU
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from datetime import date
import datetime
crypt = ['BTC','ETH']
def normalise_min_max(df): # 0-1
    return (df.min())['close'],(df.max()-df.min())['close'],(df - df.min()) / (df.max() - df.min())
def extract_window_data(df, window_len=5, f_days = 10,zero_base=True,output = False,):
    if not output:
      window_data = []
      for idx in range(len(df) - window_len + 1):
          tmp = df[idx: (idx + window_len)].copy()
          # print(tmp.index[0],tmp.index[-1])
          if zero_base:
              tmp = normalise_zero_base(tmp)
          window_data.append(tmp.values)
      return np.array(window_data)
    else:
      window_data = []
    #   print(len(df)-window_len)
      for idx in range(window_len,len(df) - f_days + 1):
          tmp = df[idx: (idx + f_days)].copy()
        #   print(tmp.index[0],tmp.index[-1])
          if zero_base:
              tmp = normalise_zero_base(tmp)
          window_data.append(tmp.values.T[0])
      return np.array(window_data)
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
def prepare_data(df, target_col, window_len=10, f_days = 10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, f_days = f_days, zero_base = zero_base,)
    X_test = extract_window_data(test_data, window_len, f_days = f_days, zero_base = zero_base,)
    y_train = extract_window_data(train_data, window_len, f_days = f_days, zero_base = zero_base,output=True)
    y_test = extract_window_data(test_data, window_len, f_days = f_days, zero_base = zero_base, output=True)
    return train_data, test_data, X_train, X_test, y_train, y_test
def build_lstm_model(input_data, output_size, neurons=100, activ_func='sigmoid', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(LeakyReLU())
    # model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model
np.random.seed(42)
window_len = 100
f_days = 10
test_size = 0.2
zero_base = False
lstm_neurons = 100
loss = 'mse'
dropout = 0.2
optimizer = 'adam'
for i in crypt:
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym={}&tsym=USD&limit=2000'.format(i))
    # res = requests.get(endpoint + '?fsym=BTC&tsym=USD&allData')
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    target_col = 'close'
    min, max, hist_norm = normalise_min_max(hist)
    _,data,_,pred_data,_,_ = prepare_data(
        hist_norm, target_col, window_len=window_len, f_days = f_days, zero_base=zero_base, test_size=test_size,)
    model = build_lstm_model(
    pred_data, output_size=f_days, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
    model.load_weights('{}_daily.h5'.format(i))
    pred = model.predict(pred_data[-101:]).squeeze()
    targets = (max * data[target_col][-window_len:]) + min
    targets.to_csv('{}_100days_actual.csv'.format(i),header=None)
    preds = (max * pred.T[0][:-1]) + min
    preds = pd.Series(index=targets.index, data=preds)
    preds.to_csv('{}_100days_predicted.csv'.format(i),header=None)
    pd.Series(index=pd.date_range(start=date.today() + datetime.timedelta(days=1), end=date.today() + datetime.timedelta(days=10)), data=(max * pred[-1])+min).to_csv('{}_next_10days.csv'.format(i))