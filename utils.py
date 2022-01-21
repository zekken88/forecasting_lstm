# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:30:49 2022

@author: CPPG02619
"""

import csv
from math import ceil
import pandas as pd
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import timedelta


def importCSV(file):
    dataset = []
    with open(file, 'r') as f:
        for row in csv.DictReader(f):
            dataset.append(row)
    df = pd.DataFrame(dataset)
    df['Date']=pd.to_datetime(df['Date'], infer_datetime_format= True)

    return df

def makeSequence(seq, step):
    x = []
    y = []
    for i in range(len(seq)):
        end = i + step
        if end >= len(seq):
            break
        x_small = seq[i:end]
        x.append(x_small)
        y.append(seq[end])
    
    return array(x),array(y)

def makeSequenceMulti(X,Y, step):
    x = []
    y = []
    for i in range(len(X)):
        end = i + step
        if end >= len(X):
            break
        x_small = X[i:end]
        x.append(x_small)
        y.append(Y[end])
    
    return array(x),array(y)

def makeSequenceMultiPred(X,Y, step, step_out):
    x = []
    y = []
    for i in range(len(X)):
        end = i + step
        end_out = end + step_out
        if end_out >= len(Y):
            break
        x_small = X[i:end]
        x.append(x_small)
        y.append(Y[end:end_out])
    
    return array(x),array(y)

def makeTrainTest(X,Y):
    length = len(X)
    train = ceil(4/5 * length)
    x_train = X[:train]
    x_test  = X[train:]
    y_train = Y[:train]
    y_test  = Y[train:]
    return x_train,y_train, x_test, y_test

def getMSE(test, pred):
    return np.square(np.subtract(test,pred)).mean()

def plotPredTest(test, pred):
    plt.plot(test, color = 'r')
    plt.plot(pred, color = 'b')
    plt.show() 

def plotPredTestnSave(test, pred, title, path):
    plt.plot(test, color = 'r',label='test')
    plt.plot(pred, color = 'b',label='prediction')
    plt.title(title)
    plt.savefig(path)
    plt.show() 

def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def addoil(dataset):
    

    col_name = 'Crude Oil'
    url = 'https://www.eia.gov/dnav/pet/hist_xls/RBRTEw.xls'
    resp = requests.get(url)
    
    data = pd.read_excel(resp.content, sheet_name='Data 1')
    data = data.values[2:]
    oil_on_date = {}
    delta = timedelta(days=1)
    for i in data:
        date = i[0] - delta
        oil_on_date[date] = i[1]
    
    oil_col = []
    for i in dataset.values:
        try:
            oil_col.append(oil_on_date[i[0]])
        except:
            date = i[0] - timedelta(days=7)
            oil_col.append(oil_on_date[date])
            
    dataset[col_name] = oil_col
    
    columns = [
               #'Date',
               'Phosphate rock',
               'Asam Sulfat',
               'Sulphur',
               'DAP',
               'Crude Oil',
               'Asam Fosfat'
               ]
    return dataset
    

file = 'Data/Sample 1 per januari 2022.csv'
dataset = importCSV(file)
