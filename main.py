import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_data(file):
    df = pd.read_csv(file)
    # print(df.head())
    return df

def preprocess_data(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df.sort_values(by='Order Date', inplace=True)  # Sort by 'Order Date'
    # df.set_index('Order Date', inplace=True)

    # df.sort_index(inplace=True)
    df.drop(columns=['Row ID'], inplace=True)
    # df.reset_index(inplace=True)
    df['Order Month'] = df['Order Date'].dt.to_period('M')
    print(df.head())
    df['Order Month'] = df['Order Month'].dt.strftime('%Y-%m')
    # df['Biannual Date'] = df['Order Date'].groupby(pd.Grouper(key='Order Date', freq='6M')).transform('min')
    df['Biannual Date'] = df['Order Date'] - pd.DateOffset(months=df['Order Date'].dt.month % 6)
    show_unique_values(df['Biannual Date'])
    # sorted_df = df.sort_values(by='Order Date', ascending=True)
    
    return df
    

def split_by_region(df):
    # Group the DataFrame by the 'Region' column
    grouped = df.groupby('Region')

    # Create an empty dictionary to store the resulting DataFrames
    region_dfs = {}

    # Iterate over each group and store the corresponding DataFrame in the dictionary
    for region, data in grouped:
        region_dfs[region] = data

    # Access the individual DataFrames using the region name as the key
    south_df = region_dfs['South']
    west_df = region_dfs['West']
    central_df = region_dfs['Central']
    east_df = region_dfs['East']

    return south_df, west_df, central_df, east_df

# Function to return a list of the unique values of the column passed in as input
def show_unique_values(df_column):
    unique_values = df_column.unique()
    print(len(unique_values))

# Further splits the region dataframes into dataframes grouped by category
def split_region_by_category(region_df, df_column, region_name):

    # grouping the data in the region df by the column that was passed in as an argument
    grouped = region_df.groupby(f'{df_column}')

    # Creating dictionary for 
    region_name = {}

    for column, data in grouped:
        variable_name = f'{column}_{region_df["Region"].iloc[0]}'
        region_name[variable_name] = data
        trend_spotting(data)
    
    print(region_name)

    return region_name



# Following methods are for the time-series prediction model

# Creating Additional features based on time series index
def create_features(df):
    df = df.copy()
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df

# Creating lag features
def create_lags(df):
    target_map = df['Sales'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('182 days')).map(target_map) # 182 days is approx 6 months and is perfectly divisible by 7 so the day of the week should match up every time.
    df['lag2'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('546 days')).map(target_map)
    return df      

def get_features():
    features = ['quarter', 'month', 'year',
                    'lag1', 'lag2', 'lag3']
    return features

def get_target():
    target = 'Sales'
    return target

def cross_validation(df):
    tss = TimeSeriesSplit(n_splits=5, test_size=182, gap=30) # Creating a TimeSeriesSplit with the number of splits being 5, the test size being 6 months and a gap of 30 days
    df = df.sort_index() # Making sure the dataframe is sorted by index

    predictions = []
    scores = []

    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]
        
        train = create_features(train)
        test = create_features(test)

        features = get_features()    
        target = get_target()

        X_train = train[features]
        y_train = train[target]

        X_test = test[features]
        y_test = test[target]

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', # setting the base prediction score and specifying the use of the tree-based models in XGBoost
                            n_estimators=500, # number of iterations in the training process
                            early_stopping_rounds=50, # Enables early stopping to prevent overfitting
                            objective='reg:linear', # Objective is to minimise the MSE for regression
                            max_depth=3, learning_rate=0.01) # Setting the max depth of each tree and setting the step size of each iteration
        
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)

        y_pred = reg.predict(X_test)
        predictions.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)
    print(df.head())

def retrain(df):
    df = create_features(df)

    features = get_features()    
    target = get_target()

    X_all = df[features]
    y_all = df[target]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', # setting the base prediction score and specifying the use of the tree-based models in XGBoost
                            n_estimators=500, # number of iterations in the training process
                            early_stopping_rounds=50, # Enables early stopping to prevent overfitting
                            objective='reg:linear', # Objective is to minimise the MSE for regression
                            max_depth=3, learning_rate=0.01) # Setting the max depth of each tree and setting the step size of each iteration
    
    reg.fit(X_all, y_all,
            eval_set=[(X_all, y_all)],
            verbose=100)
    
    reg.save_model('time_series_prediction_model.json') # Saves the retrained model as a JSON file for future prediction without retraining every time

def time_series_prediction():
    start_date = df.index.max() # the last date of the dataset - will be the start date of the prediction
    end_date = start_date + pd.Timedelta('182 days') # Creates end-date of prediction, 6 months after start date
    future  = pd.date_range(start_date, end_date, freq='B') # Creates the date range of the prediction with the interval set to 1 day
    future_df = pd.DataFrame(index=future) #Creates a future dataframe where the future date_range is the index
    future_df['isFuture'] = True # Creates 'isFuture' column to differentiate between the actual data and the future predictions
    df['isFuture'] = False
    df_with_future = pd.concat([df, future_df]) # Combines the two dataframes
    df_with_future = create_features(df_with_future) # Adds the features to the new dataframe
    df_with_future = create_lags(df_with_future) # Adds the lag features to the new dataframe

    future_and_features = df_with_future.query('isFuture').copy()

    reg = xgb.XGBRegressor() # Instantiates XGB regressor
    reg.load_model('time_series_prediction_model.json') # Loads our retrained regressor

    features = get_features()
    future_and_features['prediction'] = reg.predict(future_and_features[features])
    future_and_features['prediction'].plot(figsize=(10, 5),
                                           color='r', 
                                           title='Future Prediction')
    plt.show()


def trend_spotting(df):
    date = df['Biannual Date']
    sales = df['Sales']
    region = df.at[df.index[0], 'Region']
    category = df.at[df.index[0], 'Category']

    plt.plot(date, sales, label=f'{region}_{category}')

    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend(loc='upper left')
    plt.show()

def predict():
      
    
    
    
    return
if __name__ == '__main__':
    df = read_data('train.csv') # Reads the csv file
    print(df.corr()) 
    # preprocess_data(df)         # Sorts the file into date-time and sets the index
    #south_df, west_df, central_df, east_df = split_by_region(df) # splits the csv into smaller dataframes based on region
    #split_region_by_category(south_df, 'Category', 'South') # further splits each region by category of product
    # split_region_by_category(central_df, 'Category', 'Central')

