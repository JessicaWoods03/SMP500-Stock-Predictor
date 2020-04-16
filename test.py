import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2009, 1, 1)
end = dt.datetime(2016, 12, 31)
print('reading in stockpriceslatest...')
#shortened csv file to just Ebay prices, was too large-
# used command line grep AAPL (or any other company) stock_prices_lastest.csv >> stock_prices_ebay.cs.
# used grep EBAY stock_prices_lastest.csv > stock_prices_ebay.csv to create ebay csv file.
df = pd.read_csv('stock_prices_ebay.csv', parse_dates=True, index_col=0)
print('done.')

#print(df[['open','high']].head())
#print(df[['open','high','close_adjusted']].tail())


#print(df)
#df['Adj Close'].plot()
#plt.show()

#these are parameters we are adding in video 3
print('rolling...')
df['100movingAverage'] = df['close_adjusted'].rolling(window=100, min_periods=0).mean()
df.dropna(inplace=True) #Modifying that dataframe in place
print(df.tail())
#Graphing with just matplot lib uses axes = ax1 (you have one) ax2...ax3
#6,1 = means 6 rows and 1 column, 0,0= starting point
print('plottin...')
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1) #starting is at 5,0, 
#sharex= gives there own x axis
ax1.plot(df.index, df['close_adjusted'])
ax1.plot(df.index, df['100movingAverage'])
ax2.bar(df.index, df['volume'])
plt.show()   #shows you what you did with the matplotlib

