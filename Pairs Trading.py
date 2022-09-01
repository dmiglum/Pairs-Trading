
# Pairs Trading

### Importing Libraries
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint
# setting the seed for the random number generator
np.random.seed(107)

import matplotlib.pyplot as plt
import yfinance as yf
pd.set_option('display.max_columns', 10)


### Generating 2 fake securities
# Daily Returns
Xreturns = np.random.normal(0, 1, 100)
# sum them and shift all the prices up
X = pd.Series(np.cumsum(Xreturns), name = 'X') + 50
X.plot(figsize = (15, 7))
plt.show()

'''
Now we generate Y. Y is supposed to have a deep economic link to X, so the price of Y should vary pretty similarly. 
We model this by taking X, shifting it up and adding some random noise drawn from a normal distribution.
'''
noise = np.random.normal(0, 1, 100)
Y = X + 5 + noise
Y.name = 'Y'
pd.concat([X, Y], axis = 1).plot(figsize = (15, 7))
plt.show()


### Cointegration
'''
If two series are cointegrated, the ratio between them will vary around a mean. For pairs trading to work between two timeseries, 
the expected value of the ratio over time must converge to the mean, i.e. they should be cointegrated. 
The time series we constructued above are cointegrated.
'''
(Y/X).plot(figsize=(15,7)) 
plt.axhline((Y/X).mean(), color = 'red', linestyle = '--')
plt.xlabel('Time')
plt.legend(['Price Ratio', 'Mean'])
plt.show()


### Testing for Cointegration
'''
Statsmodels.tsa.stattools contains a cointegration test. We should see a very low p-value, as we've artificially 
created two series that are as cointegrated as physically possible.
'''
# compute the p-value of the cointegration test, which will inform us as to whether the ratio between the 2 timeseries is stationary around its mean
score, pvalue, _ = coint(X,Y)


# Using Data to find cointegrated pairs (among tech stocks)
'''
You want to choose securities you suspect may be cointegrated and perform a statistical test. 
If you just run statistical tests over all pairs, you’ll fall prey to multiple comparison bias.
'''
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.zeros((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

from datetime import datetime

# Downloading Adj Close prices across S&P and 9 tech stocks
tickers = ['SPY','AAPL','ADBE','EBAY','MSFT','QCOM', 'HPQ','NVDA','AMD','IBM']
data = yf.download(tickers, start = '2011-01-05', end = '2021-01-05')['Adj Close']

scores, pvalues, pairs = find_cointegrated_pairs(data)

# Heatmap to show the p-values of the cointegration test between each pair of stocks
import seaborn
m = [0,0.2,0.4,0.6,0.8,1]
seaborn.heatmap(pvalues, xticklabels=tickers, 
                yticklabels=tickers, cmap='RdYlGn_r', mask = (pvalues >= 0.98) )
plt.show()
print(pairs)

# Taking and ADBE and MSFT prices which appear to be cointegrated
S1 = data['ADBE']
S2 = data['MSFT']
score, pvalue, _ = coint(S1, S2)
print(pvalue)
ratios = S1 / S2
ratios.plot(figsize=(15,7))
plt.axhline(ratios.mean())
plt.legend(['Price Ratio'])
plt.show()

# Calculating Z-Score and adding it to the plot
def zscore(series):
    return (series - series.mean()) / np.std(series)

zscore(ratios).plot(figsize=(15,7))
plt.axhline(zscore(ratios).mean(), color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Ratio z-score', 'Mean', '+1', '-1'])
plt.show()


### Implementing a simple strategy of going Long the ratio when Z-Score is below -1.0, going short when z-score is above 1.0,
# and exiting positions when z-score approaches zero
'''
Instead of using ratio values, we'll use 5d Moving Average to compute to z-score, and the 60d Moving Average and 60d Standard Deviation
as the mean and standard deviation.
First, we'll break data into training set of 7 years and test set of 3 years
'''
ratios = data['ADBE'] / data['MSFT']
train_length = int(len(ratios)*0.7)
train = ratios[: train_length]
test = ratios[train_length :]

ratios_mavg5 = train.rolling(window = 5, center = False).mean()
ratios_mavg60 = train.rolling(window=60, center=False).mean()
std_60 = train.rolling(window=60, center=False).std()

zscore_60_5 = (ratios_mavg5 - ratios_mavg60) / std_60
plt.figure(figsize=(15,7))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)

plt.legend(['Ratio','5d Ratio MA', '60d Ratio MA'])

plt.ylabel('Ratio')
plt.show()

# Plotting rolling Z-Score
plt.figure(figsize=(15,7))
zscore_60_5.plot()
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-score', 'Mean', '+1', '-1'])
plt.show()


### Plotting ratios and buy/sell signals from z-score
plt.figure(figsize=(15,7))

train[60:].plot()
buy = train.copy()
sell = train.copy()
buy[zscore_60_5>-1] = 0
sell[zscore_60_5<1] = 0
buy[60:].plot(color='g', linestyle='None', marker='^')
sell[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,ratios.min(),ratios.max()))
plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
plt.show()


# Plotting prices and buy/sell signals from z-score
plt.figure(figsize=(18,9))
S1 = data['ADBE'].iloc[:train_length]
S2 = data['MSFT'].iloc[:train_length]

S1[60:].plot(color='b')
S2[60:].plot(color='c')
buyR = 0*S1.copy()
sellR = 0*S1.copy()

# When buying the ratio, buy S1 and sell S2
buyR[buy!=0] = S1[buy!=0]
sellR[buy!=0] = S2[buy!=0]
# When selling the ratio, sell S1 and buy S2 
buyR[sell!=0] = S2[sell!=0]
sellR[sell!=0] = S1[sell!=0]

buyR[60:].plot(color='g', linestyle='None', marker='^')
sellR[60:].plot(color='r', linestyle='None', marker='^')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,min(S1.min(),S2.min()),max(S1.max(),S2.max())))

plt.legend(['ADBE','MSFT', 'Buy Signal', 'Sell Signal'])
plt.show()

### Writing a simple backtest to calculate pnl (we buy 1 ratio (buy 1 ADBE stock and sell ratio x MSFT stock) when ratio is low and
# sell 1 ratio (sell 1 ADBE stock and buy ratio x MSFT stock) when it’s high)

def trade(S1, S2, window1, window2):
    # exit if either window length is 0
    if (window1 == 0) or (window2 == 0):
        return 0
    # compute rolling mean and rolling standard deviation
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    # start with no money and no position
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # sell short if the z-score is > 1
        if zscore[i] > 1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
        # buy long if the z-score is < 1
        elif zscore[i] < -1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
        # clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.5:
            money += countS1*S1[i] + countS2* S2[i]
            countS1 = 0
            countS2 = 0
    return money

trade(data['ADBE'].iloc[: train_length], data['MSFT'].iloc[:train_length], 5, 60)
# we can optimize the strategy by changing enter/exit parameters as well as moving average windows

# checking strategy on the test set
trade(data['ADBE'].iloc[train_length:], data['MSFT'].iloc[train_length:], 5, 60)

### Finding best window 2 parameter
length_scores = [trade(data['ADBE'].iloc[: train_length], data['MSFT'].iloc[:train_length], 5, l) for l in range(20, 200)]
best_length = length_scores.index(max(length_scores)) # can also do: np.argmax(length_scores)
print('best window length', best_length) # 175

# finding best window in testing data and comparing it to the window from training data
length_scores2 = [trade(data['ADBE'].iloc[train_length:], data['MSFT'].iloc[train_length:], 5, l) for l in range(20, 200)]
best_length2 = length_scores2.index(max(length_scores2)) # can also do: np.argmax(length_scores)
print('best test data window length', best_length2) # 174, close to 175 from training data

# plostting pnl by window length for train and test data
plt.figure(figsize=(15,7))
plt.plot(length_scores)
plt.plot(length_scores2)
plt.xlabel('Window length')
plt.ylabel('Score')
plt.legend(['Train', 'Test'])
plt.show()

