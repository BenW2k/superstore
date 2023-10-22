# Import dependencies
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

def read_data(file):
    df = pd.read_csv(file)
    print(df.head())
    return df

def find_unique(df):
    unique_values = df['Category '].unique()
    print(unique_values)

def preprocess_data(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

def add_monthly_column(df):
    df['Order Month'] = df['Order Date'].dt.to_period('M')


def visualisation(df):
    plt.figure(figsize=(15,5))
    plt.plot(df['Order Month'], df['Sales'])

if __name__ == '__main__':
    df = read_data('train.csv')
    preprocess_data(df)
    #print(df['Sales'].dtype)
    # add_monthly_column(df)
    # visualisation(df)
    find_unique(df)
    
