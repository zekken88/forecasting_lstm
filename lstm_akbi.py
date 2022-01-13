from typing import Sequence
import streamlit as st
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime


def lstm_proses(indexdf,slider,colom):
    data = indexdf.filter([colom])
    dataset= data.values
    training_data_len=math.ceil(len(dataset)*.6)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len,:]
    x_train = []
    y_train = [] 

    for i in range (60,len(train_data)):
      x_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i,0])

    x_train,y_train=np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

    #Membangun Model LSTM
    model = Sequential()
    model.add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(64,return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Melatih Model
    model.fit(x_train, y_train,batch_size=5,epochs=50)

    test_data= scaled_data[training_data_len-60: , :]
    x_test=[]
    y_test= dataset[training_data_len: , :]
    for i in range(60, len(test_data)):
      x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(predictions-y_test)**2)

    #hasildari model
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions']=predictions
    result=pd.concat([indexdf,valid['Predictions']], axis=1)
    #Prediksi kedepen
    slider_mines=slider
    slider_mines*=-1
    new_df=df.drop('Date',1)
    last_60_days = new_df[-60:]
    for i in range(slider):
      last_60_days=new_df[-60:]  
      last_60_days_array=np.reshape(last_60_days,(-1,1))
      last_60_days_scaled= scaler.transform(last_60_days_array)
      X_test=[]
      X_test.append(last_60_days_scaled)
      X_test = np.array(X_test)
      X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
      pred_price=model.predict(X_test)
      pred_price = scaler.inverse_transform(pred_price)
      new_df=np.append(new_df,pred_price)
      
    future_pred=new_df[slider_mines:]

    df_date=df[['Date']]
    last_date=df_date[-1:]
    future_date=[]
    i=7

    for x in range (slider):
      end_date = last_date + datetime.timedelta(days=i)
      future_date.append(end_date)
      i+=7

    future_date=np.array(future_date)
    future_date=future_date.reshape(-1)
    future_result={'Date':future_date,'Future Prediction':future_pred}
    final_df=pd.DataFrame(data=future_result)
    final_result=pd.concat([df,final_df],axis=0)
    final_result['Date']=pd.to_datetime(final_result['Date'], infer_datetime_format= True)
    final_result=final_result[['Date','Future Prediction']]
    final_result=final_result[slider_mines:]
    final_result=final_result.set_index('Date')

    fix_result=pd.concat([result,final_result],axis=0)

    return rmse,result,fix_result

st.write("""
# Forecasting Data Bahan Baku Pupuk Dengan Metode LSTM
PT. Petrokimia Gresik
""")
st.sidebar.header('User Input Features')

uploaded_file = st.sidebar.file_uploader("Upload your input csv file", type=["csv"])

if uploaded_file is not None :
  df = pd.read_csv(uploaded_file)
  st.dataframe(df)
  pilihan=df.columns
  pilihan=pilihan.drop('Date',1)
  angka=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
  colom=st.sidebar.selectbox('Pilih data yang akan diolah',  pilihan)
  slider=st.sidebar.selectbox('Jumlah Minggu yang ingin diprediksi',angka)
  if slider != 0:
    df=df[['Date', colom]]
    df.dropna(inplace=True)
    Jumlah_null=df.isnull().sum()
    st.subheader("Jumlah Data")
    st.write(df.shape)
    df['Date']=pd.to_datetime(df['Date'], infer_datetime_format= True)
    indexdf=df.set_index(['Date'])
    st.subheader("Chart Data")
    st.line_chart(indexdf)
    st.dataframe(indexdf)
    rmse,result,fix_result=lstm_proses(indexdf,slider,colom)
    st.subheader("Root Mean Square ERROR!")
    st.write(rmse)
    st.subheader("Hasil Model")
    st.line_chart(result)
    st.write(result)
    st.subheader("Hasil Forecasting")
    st.line_chart(fix_result)
    st.write(fix_result)
    result_csv=fix_result.to_csv().encode('UTF-8')
    st.download_button(
      label="Download data as CSV",
      data=result_csv,
      file_name='Hasil.csv',
      mime='text/csv',
    )