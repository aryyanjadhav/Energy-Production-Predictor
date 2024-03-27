#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np 
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


# In[4]:


df=pd.read_csv("ep.csv",delimiter=';')


# In[5]:


#df


# In[7]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[6]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)






# In[18]:


import pickle


# In[20]:
def predict(temperature, exhaust_vacuum, amb_pressure, r_humidity):
    rf_model = RandomForestRegressor()
    model_path='/Users/aryan/Desktop/rf_model.pkl'
    with open(model_path, 'rb') as file:
        rf_model = pickle.load(file)

    input_data = [[temperature, exhaust_vacuum, amb_pressure, r_humidity]]

    prediction = rf_model.predict(input_data)
    return prediction[0]


def main():
    st.title('Energy Production Prediction')
    st.sidebar.subheader("User Input Features")
    def user_input_features():
        temperature=st.sidebar.number_input("Temperature", min_value=0.0, max_value=100.0,value=25.0,key="temperature")
        exhaust_vacuum=st.sidebar.number_input("Exhaust Vacuum", min_value=0.0, max_value=100.0,value=50.0,key="exhaust_vacuum")
        amb_pressure=st.sidebar.number_input("Ambient Pressure", min_value=0.0, max_value=1200.0,value=75.0,key="amb_pressure")
        r_humidity=st.sidebar.number_input("R_humidity", min_value=0.0, max_value=500.0,value=50.0,key="r_humidity")
        return temperature, exhaust_vacuum, amb_pressure, r_humidity
    temperature, exhaust_vacuum, amb_pressure, r_humidity =user_input_features()
    result =""
    if st.button("predict"):
        result=predict(temperature,exhaust_vacuum,amb_pressure,r_humidity)
    st.success("The Predicted Value is {}".format(result))
    
if __name__=='__main__':
    main()


# In[ ]:




