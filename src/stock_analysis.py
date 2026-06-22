"""
Dennis Toups
CSCI 4740-271I
Final Project
Dr. Ahmed Shaffie
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
#import yfinance as yf
#yf.shared._DFS.clear()      # clears cached ticker objects
#yf.shared._ERRORS.clear()   # clears error cache

######## Function Definitions #################################################

"""------ Confusion Matrix Calculations ---------------------------------------
Functionality: Computes metrics from confusion matrix and displays results

Inputs: Confusion matrix

Outputs: Prints statistics
"""

def cm_metrics(conf_matrix):
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Display results
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}\n")
    
    return 

"""----- 80/20 ----------------------------------------------------------------
Functionality: Performs 80/20 test/train split for both naive Bayes and
decision tree classifiers

Inputs: Feature and target data in list format
        
Outputs: 
    - Decision Tree results
    - Bayesian inference results
    - Confusion matrix results
"""

def Eighty_Twenty(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, \
                                                        random_state=42)
    # Train model, predict
    clf_dt = DecisionTreeClassifier(criterion='entropy')
    clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    
    # Create confusion matrix
    cm_dt = confusion_matrix(y_test, y_pred)
    
    accuracy_1 = accuracy_score(y_test, y_pred)
    print("Decision Tree: ")
    print(f"Accuracy: {accuracy_1:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Train model, predict
    clf_b = GaussianNB()
    clf_b.fit(X_train, y_train)
    y_pred = clf_b.predict(X_test)
        
    # Create confusion matrix
    cm_b = confusion_matrix(y_test, y_pred)
    
    accuracy_2 = accuracy_score(y_test, y_pred)
    print("Naive Bayes: ")
    print(f"Accuracy: {accuracy_2:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf_dt, cm_dt, clf_b, cm_b

"""----- Preprocess dataset ---------------------------------------------------
Functionality: Takes raw data and normalizes by calculating skewness and 
relative volume, and builds predicted price trend.

Inputs: 
    - raw - raw data to be analyzed
    - thresh - threshold value to classify stock price trend as upward or not
    - block - total length of one period, in days,  of calculation and analysis
        
Outputs: 
    - data - list containing skew, relative volume, and price trend
"""

def preprocess(raw, thresh, block):
    data = []
    blk = block-1 # Need an end reference for pandas series which begin at 0
    
    # Loop by day, but analyze by block
    for idx in range( blk, len(raw) ):
        
        # Slice a data block from the raw data
        raw_block = raw.iloc[idx - blk : idx ]
        
        # Compute rolling average closing price
        avg_price = float(raw_block['Close'].mean().item())
        
        # Normalize by dividing raw prices by average price for the period
        norm_price = raw_block['Close'] / avg_price
        
        # Need the raw last closing price for comparison with moving average 
        # to classify price trend
        last_price = float(raw_block['Close'].iloc[-1].item())
        
        # Calculate skewness
        skew = float(norm_price.skew().item())
        
        # Compute rolling average trading volume
        avg_vol = float(raw_block['Volume'].mean().item())
        
        # Normalize by dividing trading volume by average volume of the period
        norm_vol = raw_block['Volume'] / avg_vol
        today_rel_vol = float(norm_vol.iloc[-1].item()) # Today's volume
        
        # Classify trend (period close > threshold * moving average = 
        # upward trend)
        if last_price > thresh * avg_price:
            trend = 1 # Trending upward
            
        else:
            trend = 0 # Not trending upward
        
        trend = int(trend)
        
        data.append([skew, today_rel_vol, trend])
        
    return np.array(data)

"""----- Analyze --------------------------------------------------------------
Functionality: Performs initial check for missing or invalid data and conducts
both decision tree and Bayesian inference analysis on a set of raw stock data 
using various values of stock price rise thresholds and for different numbers 
of evaluation periods.

Inputs: Individual stock price raw data set and ticker symbol
        
Outputs:
    - Prints decision tree visualization
    - Sub-functions will print confusion matrix information

Function Calls: 
    - data_calc() to preprocess the data for analysis
    - cm_metrics() to print confusion matrices
"""

def analyze(tkr_data, symbol):
    
    # Check the data for missing or NAN values:
    if tkr_data.isna().any().any():
        return 'Missing or invalid data'
    
    # Loop through different stock price rise threshold (upward trend) values
    # and moving average period analysis values
    for rise_thresh in THRESH:
            
        for period in PERIOD:
            data_1 = preprocess(tkr_data, rise_thresh, period)
            X = [row[:2] for row in data_1]
            X = np.array(X)
            y = [row[2] for row in data_1]
            y = np.array(y)
            
            dtclf_8020, dtcm_8020, bclf_8020, bcm_8020 = Eighty_Twenty(X, y)
            
            print(f'Symbol: {symbol}***************\nAnalysis Period (Days): '\
                  , period, '\nStock Upward Trend Threshold: ', \
                      (int(100*(rise_thresh-1))), '%\n')
                
            print('Decision Tree:')
            cm_metrics(dtcm_8020)
            
            print('Bayesian Inference:')
            cm_metrics(bcm_8020)
            print('\n')

            # Visualize the Decision Tree
            plot_tree(dtclf_8020, filled=True, feature_names=["Skewness", \
              "Volume"], class_names=["Price Not Increasing", \
                    "Price Increasing"])
            plt.title(f'Decision Tree {symbol} {period} days {rise_thresh} '
                      'trend threshold')
            plt.show()

    return

######## Script Execution #####################################################

"""
# Download stock data
for ticker in ["FSM", "RIG"]:
    df = yf.download(ticker, start="2020-11-25", end="2025-11-25", \
                     auto_adjust=True)
    df.to_pickle(f"{ticker}_daily.pkl")# Create FSM_daily.pkl and RIG_daily.pkl
    #num_syms += 1
"""    

# Raw stock data
t_1 = pd.read_pickle("FSM_daily.pkl") # Ticker 1
t_2 = pd.read_pickle("RIG_daily.pkl") # Ticker 2

# Set intervals
PERIOD = (14, 60)
THRESH = (1.05, 1.10)
# Execute for two symbols
analyze(t_1, 'FSM')
    
analyze(t_2, 'RIG')
