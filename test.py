import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import random
from sklearn.model_selection import train_test_split # Split data
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor # Random Forest Classifier
from sklearn.model_selection import train_test_split
import numpy as np

data_btc = pd.read_csv('Binance_BTCUSDT_d.csv')
df = data_btc

df = df.drop(columns=['Date'])
df = df.drop(columns=['Volume USDT'])
#def xulydulieu():
    #data_btc = pd.read_csv('Binance_BTCUSDT_d.csv')
    #data_btc.head(5)
    #df.drop('Date')
    #df.drop(columns=['Date'])
    #global df
    #df = df.drop(columns=['Date'])
    #df = df.drop(columns=['Volume USDT'])
    #print("Is There any 'NaN' value: ", df.isnull().values.any())
    #print("Is there any duplicate value: ", df.index.duplicated().any())
    #print("Historical Data Shape: ", df.shape)
    #print(df.head(5))
def vebieudo(price_predict, price_real):
    ngay = [0 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ]
    price_predict1 = price_predict
    price_real1 = price_real
    plt.plot(ngay, price_predict1, color = 'red',label = 'Predict BTC Price')
    plt.plot(ngay, price_real1, color='blue', label = 'Real BTC Price')
    plt.xlabel('Thoi Gian')
    plt.ylabel('Gia USDT')
    plt.title('Bitcoin Price Prediction')
    plt.show()
def  chon7ngay(t,m_tree, n_try) :
    global df
    #xulydulieu()
    historical_df = df
    for i in range(1, t):  # for 7 days
        historical_df["Open_b_" + str(i)] = df['Open'].shift(i)
        historical_df["High_b_" + str(i)] = df['High'].shift(i)
        historical_df["Low_b_" + str(i)] = df['Low'].shift(i)
        historical_df["Close_b_" + str(i)] = df['Close'].shift(i)
        historical_df["Volume_(BTC)_b_" + str(i)] = df['Volume BTC'].shift(i)
        historical_df["Volume_(Currency)_b_" + str(i)] = df['Volume USDT mil'].shift(i)

    historical_df = historical_df.dropna()  # drop the first rows. They don't have previous information
    #print("Historical Data Shape: ", historical_df.shape)
    df_labels = df['Close']
    df_labels = df_labels.dropna()
    #df_labels = df_labels.drop()
    y = df_labels[:(863-t+1)]
    X = historical_df[1:]
    X = X.reset_index()
    X = X.drop(columns=['index'])
    #print("Historical Data X Shape: ", X.shape)
    #print(x.head(5))
    #xoz = X[math.floor((X.index)/2)]
    #print(xoz.head(3))
    X.to_csv("historical.csv")
    #print("y Data Shape: ", y.shape)
    #print("y Data Shape: ", y.head(5))
    #df_labels.drop(2,axis = 0)
    y.to_csv("df_labels.csv")
    #df_labels2 = df_labels['Close'].shift[-1]
    #print(df_labels.head(5))
    #print(historical_df.head(3))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    prediction_days = 30
    X_train = X[prediction_days:]
    X_test = X[:prediction_days]
    y_train = y[prediction_days:]
    y_test = y[:prediction_days]
    #print("X_train.head(3) ", X_train.head(3))
    #print("y_train.head(3) ", y_train.head(3))
    #print("X_train.head(3) ", X_test.head(3))
    #print("y_test.head(3) ", y_test.head(3))
    #trainingAndTestData(X_train, y_train, X_test, y_test)
    # global X_train, y_train, X_test, y_test
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=m_tree, max_depth=n_try, random_state=42, min_samples_split=5)
    # Train the model on training data
    rf.fit(X_train, y_train)
    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    #print('Mape: ', mape)
    #print(errors)
    #print(predictions)
    #print(y_test)1
    ###VE BIEU DO#######
    vebieudo(predictions, y_test)
    ###########################
    return round(accuracy, 2)
    # print(errors)
    # get_true= round(accuracy,2)
    # return get_true
    #print(predictions)
    #print(y_test)
    # print(X_train.head(3))
def trainingAndTestData(X_train, y_train, X_test, y_test):
    #global X_train, y_train, X_test, y_test
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    # Train the model on training data
    rf.fit(X_train, y_train);
    # Use the forest's predict method on the test data
    predictions = rf.predict(X_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    #print('Accuracy:', round(accuracy, 2), '%.')
    #print(errors)
    #get_true= round(accuracy,2)
    #return get_true
    #print(predictions)
    #print(y_test)
def findAccuracyMax():
    maxx = chon7ngay(2)
    kqi = 2
    for i in range(2,15):
        kq = chon7ngay(i)
        if kq>maxx:
            maxx = max(kq,maxx)
            kqi = i
        print( "ket qua thu " ,format(i), "la ", kq)
    print("ket qua maxx la: ", maxx, "tai w bang ", kqi)

chon7ngay(2,15,7)
def findAccuracyMaxMtreeNtry():
    maxx = 0
    n_tree = 0
    m_try = 0
    kq = 0
    for i in range(5, 30, 2):
        for j in range(3,10,1):
            kq = chon7ngay(2,i,j)
            if kq>maxx:
                maxx = kq
                n_tree = i
                m_try = j
                print("mtree", i, "ntry", j, "KQ: ", kq)

            #print("mtree", i, "ntry", j, "KQ: ", kq)
    print("KQ la: ", maxx, "voi n_tree bang ", n_tree, " va m_try bang: ", m_try)
#findAccuracyMax()
#findAccuracyMaxMtreeNtry()

#vebieudo()