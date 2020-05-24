# -*- coding: utf-8 -*-
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
class Preprocessor:
    
    def __init__(self):
        print("Preprocessor initialized")
    def manageMissingValues(self,df, defaultvalue, manage):
        if manage =='replace':
            df.fillna(defaultvalue)
        elif manage =='interpolate':
            df.interpolate(method ='linear', limit_direction ='forward')        
        elif manage == 'drop':
            df.dropna(axis=0)  # remove the fields
        else:
            raise ValueError('Not supported operation for missing values') 
        return df 
        
    def  scale (self, df, type):
        #print(df.columns)
        if type=='minmax':
            scaler= MinMaxScaler()
            return (scaler.fit_transform(df))
        else:
            scaler= StandardScaler()
            return (scaler.fit_transform(df))

    def replace(self, df, old, new):
        df=df.replace(old,new)
        return df
        
         