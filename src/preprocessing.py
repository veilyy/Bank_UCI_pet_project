# Модуль создан для подготовки данных к обучению

import os
import pandas as pd
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def del_feat(df):
    #Нельзя знать длительность звонка до его начала
    df = df.drop(['duration'], axis=1)
    return df

def rare_cat(df):
    #объединяем редкие категорий

    df['default'] = df['default'].replace('yes', 'unknown')

    df['education'] = df['education'].replace('illiterate', 'unknown')

    df['marital'] = df['marital'].replace('unknown', 'single')

    return df

def OHE(df, cat_cols=None):
    
    if cat_cols is None:
        cat_cols = ['job', 'marital', 'education', 'default', 
                   'housing', 'loan', 'contact', 'month',
                   'day_of_week', 'poutcome']
    
    df = pd.get_dummies(data=df, columns=cat_cols, drop_first=True)
    
    return df

def fit_scaler(df, num_cols):

    scaler = StandardScaler()

    scaler.fit(df[num_cols])

    return scaler
    
def transform_scaler(df, scaler, num_cols):
    
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.transform(df[num_cols])

    return df_scaled

def train_test(df):
    X = df.drop(['y'], axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    return  X_train, X_test, y_train, y_test

def preprocessing(df):
# Итоговый пайплайн, возвращает словарь
    df = df.copy()

    df['y'] = df['y'].map({'yes' : 1, 'no' : 0})

    df = del_feat(df)
    
    num_cols =['age',
             'campaign',
             'pdays',
             'previous',
             'emp.var.rate',
             'cons.price.idx',
             'cons.conf.idx',
             'euribor3m',
             'nr.employed',]
    
    df = rare_cat(df)
    
    df = OHE(df)

    X_train, X_test, y_train, y_test = train_test(df)

    scaler = fit_scaler(X_train, num_cols)
    
    X_train_scaled = transform_scaler(X_train, scaler, num_cols)
    X_test_scaled = transform_scaler(X_test, scaler, num_cols)

    return {
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(X_train_scaled.columns)
    }       