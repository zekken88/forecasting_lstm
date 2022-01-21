# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:26:41 2022

@author: CPPG02619

https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

with Multiple Input Multi-Step Output
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from utils import makeSequenceMultiPred,makeTrainTest,importCSV,getMSE,plotPredTestnSave
import datetime
from os.path import join
from numpy import array
import os
x_train, y_train, x_test, y_test  = (0,0,0,0)

def createModel():
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 


def plotdata(file):
    #show plot data
    dataset = importCSV(file)
    dataset['Date']=pd.to_datetime(dataset['Date'], infer_datetime_format= True)
    indexdf=dataset.set_index(['Date'])
    indexdf = indexdf.apply(pd.to_numeric)
    plt.figure(figsize=(16,8))
    plt.title('Average price History')
    plt.plot(indexdf)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Avarage Price USD', fontsize=18)
    plt.show()


def importDataset(file):
    dataset = importCSV(file)
    dataset = dataset[columns]

    """
    dataset['Date']=pd.to_datetime(dataset['Date'], infer_datetime_format= True)
    date = datetime.datetime(2000,1,1)
    for i in range(len( dataset['Date'])):
        dataset['Date'][i] = (dataset['Date'][i]-date).days  
    """    
    
    dataset = dataset.apply(pd.to_numeric)
    return dataset

def sepparateXY(dataset,n_steps_in, n_steps_out ):
    X = dataset[columns[:-1]].values
    Y = dataset[columns[-1]].values
    x,y = makeSequenceMultiPred(X,Y, n_steps_in, n_steps_out)
    return x,y

def makePrediction(x_test):
        
    y_pred = []
    count = 0
    length = len(x_test)
    for x in x_test:
        if count %100 == 0:
            print("%s / %s" %(count, length))
        count+=1
        x = x.reshape((1, n_steps_in, n_features))
        y_pred.append(model.predict(x))
    # demonstrate prediction
    y_pred = array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[2])
    return y_pred


def createOutput(y_pred, y_test):
    title       = 'all variable epoch = %s in = %s' % (epoch, n_steps_in)
    MSE         = getMSE(y_test, y_pred)
    print(MSE)
    
    notes = ['train data : %s ' %(x_train.shape[0]),
             'test data : %s ' %(x_test.shape[0]),
             'steps_in : %s' % (n_steps_in),
             'steps_out : %s' % (n_steps_out),
             'epoch : %s' % (epoch),
             'mse pred : %s ' % (MSE)
             ]
    path = join('output3', title)
    try:
        os.mkdir(path)
    except:
        pass
    
    
    y_pred = list(y_pred)
    y_test = list(y_test)
    notes.append('')
    for y in range(len(y_pred[0])):
        
        y_pred_plot = [i[y] for i in y_pred]
        y_test_plot = [i[y] for i in y_test]
        paths = join(path, "Asam Fosfat - %s" % (y+1))
        
        plotPredTestnSave(y_pred_plot,y_test_plot,  
                          'prediction week +%s'%(y+1), 
                          paths)
        mse = getMSE(y_pred_plot, y_test_plot)
        
        notes.append('prediction week+%s : %s ' % (y+1, mse))
        
        
        
        print(getMSE(y_pred_plot, y_test_plot))
        

        
    file = join(path, 'notes.txt')
    with open(file, 'w') as f:
        for i in notes:
            f.write(i)            
            f.write('\n')

columns = [
           #'Date'
           'Phosphate rock',
           'Asam Sulfat',
           'Sulphur',
           'DAP',
           'Crude Oil',
           'Asam Fosfat'
           ]

n_steps_in  = 10 #n weeks as input
n_steps_out = 5 #n weeks as output
n_features  = len(columns)
epoch       = 300
file        = join('Data','Sample 1 process.csv')


dataset     = importDataset(file)

y = dataset[columns[-1]]
y2 = y.copy()
for item in range(len(y)):
    
    if item == 0:
        y2[0] = 0
        continue
    if y[item]-y[item-1] !=0:
        y2[item] = 1
    else:
        y2[item] = 0
    
dataset[columns[-1]] = y2


x,y         = sepparateXY(dataset, n_steps_in, n_steps_out)

x_train, y_train, x_test, y_test = makeTrainTest(x,y)

n_features  = x.shape[2]
model       = createModel()
model.fit(x_train, y_train, epochs=epoch, verbose=1)
y_pred      = makePrediction(x_test)

createOutput (y_pred, y_test)


