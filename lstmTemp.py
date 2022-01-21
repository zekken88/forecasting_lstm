# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 09:26:41 2022

@author: CPPG02619
https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

with Multiple Parallel Input and Multi-Step Output


"""
<<<<<<< HEAD

=======
>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053
import tensorflow.keras.callbacks as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
from utils import split_sequences,makeTrainTest,importCSV,getMSE,plotPredTestnSave,addoil
import datetime
from os.path import join
from numpy import array
x_train, y_train, x_test, y_test  = (0,0,0,0)

import os 


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
    
def createModel():

    
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features), dropout=dropout))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(loss='mse', optimizer='adam')
    return model

def importDataset(file):
    dataset = importCSV(file)
    dataset['Date']=pd.to_datetime(dataset['Date'], infer_datetime_format= True)
    dataset = addoil(dataset)
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
    dataset     = dataset[columns]
    dataset     = dataset.values
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
    title           = '%s epoch = %s in = %s' % (tit, epoch, n_steps_in) 
    MSE         = getMSE(y_test, y_pred)
    print(MSE)
    
    notes_csv                 = {}
    notes_csv['n_steps_in']   = n_steps_in
    notes_csv['epoch']        = epoch
    
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
    
    for col in range(len(columns)):
        notes_csv[columns[col]] =  int(getMSE([i[h][col] for h in range(4) for i in y_pred],
                                          [i[h][col] for h in range(4) for i in y_test]))
    
        
    for y in range(len(y_pred[0])):
        
        y_pred_plot = [i[y] for i in y_pred]
        y_test_plot = [i[y] for i in y_test]
        paths = join(path, "all variable - %s" % (y+1))
        
        plotPredTestnSave(y_pred_plot,y_test_plot,  
                          'prediction week +%s'%(y+1), 
                          paths)
        mse = int(getMSE(y_pred_plot, y_test_plot))
        
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
<<<<<<< HEAD
    return MSE, notes_csv
=======
    return MSE
>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053

def savemodel(model):
    try:
        os.mkdir('model')
    except:
        pass
    path = join('model', 'Multiple Parallel Input and Multi-Step Output')
    model.save(path)



columns = [
           #'Date',
           'Phosphate rock',
           'Asam Sulfat',
           'Sulphur',
           'DAP',
           'Crude Oil',
           'Asam Fosfat'
           ]

n_steps_in  = 10 #n weeks as input
n_steps_out = 5 #n weeks as output
epoch       = 300
dropout     = 0
n_features  = len(columns)
file        = join('Data','Sample 1 per januari 2022.csv')
model       = 0
<<<<<<< HEAD
dataset     = []
dataset     = importDataset(file)
=======

>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053

def process():
    global model
    global x_train, y_train, x_test, y_test 
    title           = '%s epoch = %s in = %s' % (tit, epoch, n_steps_in) 
<<<<<<< HEAD
    
=======
    dataset     = importDataset(file)
>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053
    es          = EarlyStopping(monitor='loss', verbose=1, patience=epoch/10)
    mc          = ModelCheckpoint(join('model', '%s.h5'%(title)), monitor='loss', mode='min', save_best_only=True)    
    x,y         = sepparateXY(dataset, n_steps_in, n_steps_out)
    x_train, y_train, x_test, y_test = makeTrainTest(x,y)
    model       = createModel()
    model.fit(x_train, y_train, epochs=epoch, verbose=1, callbacks = [es, mc])
    y_pred      = makePrediction(x_test)
    
    return createOutput (y_pred, y_test)


#savemodel(model)

#=======================
#===== Evaluation ======
#=======================

import logging
for test in range(1,4):
<<<<<<< HEAD
    summary_filename = 'summary_test__with_notes%s'%(test)
    tit             = 'all variable test notes%s'%(test)
    eval_in         = [5,10,15]
    eval_epoch      = [100,300,500]
    mse_eval        = {i : { e: 0 for e in eval_epoch} for i in eval_in}
    notes_csv       = []
=======
    summary_filename = 'summary_test_%s'%(test)
    tit             = 'all variable test%s'%(test)
    eval_in         = [5,10,15]
    eval_epoch      = [100,300,500]
    mse_eval        = {i : { e: 0 for e in eval_epoch} for i in eval_in}
>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053
    for ins in eval_in:
        n_steps_in = ins
        for ep in eval_epoch:
            logging.warning('===================n_step_in : %s | epoch : %s======================'%(ins,ep))
<<<<<<< HEAD
            epoch = ep
            
            mse_eval[n_steps_in][epoch],notes = process()
            notes_csv.append(notes)
=======
            epoch = ep 
            mse_eval[n_steps_in][epoch] = process()
>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053
            
    summary = ["===================="]
    summary_csv = []
    for ins in mse_eval:
        for ep in mse_eval[ins]:
            summary.append('-----')    
            one_row_csv     = {}
            one_row_csv['n_steps_in']   = ins
            one_row_csv['epoch']        = ep
            one_row_csv['mse']          = int(mse_eval[ins][ep])
            summary_csv.append(one_row_csv)
            
            summary.append('n_steps_in  :%s'%(ins))
            summary.append('epoch       :%s'%(ep))
            summary.append('mse         :%s'%(int(mse_eval[ins][ep])))
            summary.append('')
            summary.append('')
<<<<<<< HEAD
            
    summary_csv += notes_csv
=======
        
>>>>>>> 31661a32aa362de73b9a9fd81d6bdfb249011053
    with open(summary_filename + '.txt', 'a') as f :
        for i in summary:
            f.write(i)
            f.write('\n')
    
    
    csv =  pd.DataFrame(summary_csv)
    csv.to_csv("%s.csv"%(summary_filename))
    
        
