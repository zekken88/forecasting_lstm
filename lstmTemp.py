# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:26:41 2022

@author: CPPG02619
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

with Multiple Parallel Input and Multi-Step Output


"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import pandas as pd
import matplotlib.pyplot as plt
from utils import split_sequences,makeTrainTest,importCSV,getMSE,plotPredTestnSave
import datetime
from os.path import join
from numpy import array
x_train, y_train, x_test, y_test  = (0,0,0,0)
import os 

def createModel():
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    return model 


def plotdata(indexdf):
    #show plot data
    indexdf=importCSV(file).set_index(['Date'])
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
    dataset = dataset[columns]
    dataset = dataset.values
    x,y = split_sequences(dataset, n_steps_in, n_steps_out)
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
    y_pred = y_pred.reshape(y_pred.shape[0],y_pred.shape[2],y_pred.shape[3])
    return y_pred

def createOutput(y_pred, y_test):
    title       = 'without Asam Fosfat epoch = %s in = %s' % (epoch, n_steps_in)
    MSE         = getMSE(y_test, y_pred)
    print(MSE)
    
    notes = ['train data : %s ' %(x_train.shape[0]),
             'test data : %s ' %(x_test.shape[0]),
             'steps_in : %s' % (n_steps_in),
             'steps_out : %s' % (n_steps_out),
             'epoch : %s' % (epoch),
             'mse pred : %s ' % (MSE)
             ]
    path = join('output', title)
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
        paths = join(path, "all variable - %s" % (y+1))
        
        plotPredTestnSave(y_pred_plot,y_test_plot,  
                          'prediction week +%s'%(y+1), 
                          paths)
        mse = getMSE(y_pred_plot, y_test_plot)
        
        notes.append('prediction week+%s : %s ' % (y+1, mse))
        
        
        
        print(getMSE(y_pred_plot, y_test_plot))
        
    notes.append('')
    notes.append('')
    for col in range(len(columns)):
        
        notes.append('prediction for %s'%(columns[col]))
        
        for y in range(len(y_pred[0])):
            y_pred_plot = [i[y][col] for i in y_pred]
            y_test_plot = [i[y][col] for i in y_test]
            paths = join(path,  "%s - %s" % (columns[col], y+1))
            plotPredTestnSave(y_pred_plot,y_test_plot,  
                              'prediction week +%s'%(y+1), 
                              paths)
            mse = getMSE(y_pred_plot, y_test_plot)
            
            notes.append('\tprediction week+%s : %s ' % (y+1, mse))
            
            
            
            print(getMSE(y_pred_plot, y_test_plot))
        notes.append('')
        
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
           #'Asam Fosfat'
           ]

n_steps_in  = 15 #n weeks as input
n_steps_out = 5 #n weeks as output
epoch       = 500
n_features  = len(columns)
file        = join('Data','Sample 1 process.csv')


dataset     = importDataset(file)
x,y         = sepparateXY(dataset, n_steps_in, n_steps_out)

x_train, y_train, x_test, y_test = makeTrainTest(x,y)

n_features  = x.shape[2]
model       = createModel()
model.fit(x_train, y_train, epochs=epoch, verbose=1)
y_pred      = makePrediction(x_test)

createOutput (y_pred, y_test)




