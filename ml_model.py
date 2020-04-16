import numpy as np
import pandas as pd
import pickle
from collections import Counter
#these are the classifiers which is the final step in the ML model Process that our data is ran through
from sklearn import svm,model_selection,neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    hm_days = 7 #how many days out in the future the percentage change
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    #this is to go from range from 1 to 7 days
    for i in range(1, hm_days+1):
        print(i)
        #this is an algorythm that takes and predicts two days from now, divided by todays
        #price and its in percent change, its the price from two days from now - todays price, divided by todays price times 100
        df['{}_{}d'.format(ticker,i)] =(df[ticker].shift(-i)-df[ticker])/(df[ticker]) #shifting -i is shifting up by i numbers
    df.fillna(0, inplace=True)
    return tickers, df
#this is within 33% accurate, will need back testing to narrow down the accuracy
#what we are going to do is pass in future prices into this argument (args) is.
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02 # means 2%
    for col in cols:
        if col > requirement:
            return 1        #this is for sell
        if col < -requirement:
            return -1     #this for buy
    return 0   #this is for hold
    
#the next function is to map these columns
    print("process")
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    print("process..")
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data Spread:', Counter(str_vals)) #this gives us distribution to help us get how accurate our classifier is
    df.fillna(0, inplace=True)

    df = df.replace([np.inf, -np.inf], np.nan) #this is replace infinit changes to np.nan
    df.dropna(inplace=True)

    #create the feature sets and the labels, this helps get explicit future data
    df_vals = df[[ticker for ticker in tickers]].pct_change()  #normalized data
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

#the final step to the ML model is running the data throught the classifiers

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    #create training and testing our data
    X_train, X_test, y_train, y_test =model_selection.train_test_split(X, y, test_size = 0.25)

    #create a classifier

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))  #this checks if the predictions are skewd or not
    return confidence

do_ml('BAC')  #we are running ML with BAC = Bank of America
