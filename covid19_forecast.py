#%% Import OS and keras packages along with assigning Keras Backend as Tensorflow
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# Import module for data cleaning and processing and mlflow for MLOps integration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import tensorflow as tf

# Import window generator from data windowing
from data_windowing import WindowGenerator

# Get file path of train and test dataset
train_csv_path = os.path.join(os.getcwd(),r"datasets\cases_malaysia_train.csv")
test_csv_path = os.path.join(os.getcwd(),r"datasets\cases_malaysia_test.csv")

# Import train and test dataset and load into respective dataframe
train_case_df = pd.read_csv(train_csv_path)
test_case_df = pd.read_csv(test_csv_path)

# Get the date column from dataframe
train_date_time = pd.to_datetime(train_case_df.pop('date'), format="%d/%m/%Y")
test_date_time = pd.to_datetime(test_case_df.pop('date'), format="%d/%m/%Y")

# Fill in the null value from train dataset as zeros
train_case_df = train_case_df.fillna(0)

# Replace string value of question mark and space from train dataset with NaN values
train_case_df = train_case_df.replace('?', np.nan)
train_case_df = train_case_df.replace(" ", np.nan)

# Get the row indexes of train dataset with NaN values on 'cases_new' column
row_numbers = train_case_df[train_case_df['cases_new'].isnull()].index

# Loop through the row index from row indexes to perform both data cleanup and 
# feature engineering where number of new cases calculated by subtracting number 
# of active cases specific day with previous day for train dataset
for row_num in row_numbers:
    train_case_df.loc[row_num,'cases_new'] = (train_case_df.loc[row_num,'cases_active']-train_case_df.loc[row_num-1,'cases_active'])+train_case_df.loc[row_num,'cases_recovered']

# Get index with NaN value in cases_new column from test dataset
row_index = test_case_df.loc[test_case_df['cases_new'].isnull()].index
row_index = row_index[0]

# perform both data cleanup and feature engineering same method as train dataset, but for test dataset
test_case_df.loc[row_index,'cases_new'] = (test_case_df.loc[row_index,'cases_active']-test_case_df.loc[row_index-1,'cases_active'])+test_case_df.loc[row_index,'cases_recovered']

# Convert data type to float64 for both train and test dataset
train_case_df = train_case_df.astype('float64')
test_case_df = test_case_df.astype('float64')

# Get the information for displaying the graph of dataset
train_case_df_copy = train_case_df.copy()
test_case_df_copy = test_case_df.copy()
train_case_df_copy.index = train_date_time
test_case_df_copy.index = test_date_time

# Plot the train dataset
print("Train Dataset Graph")
train_case_df_copy.plot(subplots=True,figsize=(10,20))
plt.show()

# Plot the test dataset
print("Test Dataset Graph")
test_case_df_copy.plot(subplots=True,figsize=(10,20))
plt.show()

# Show there are no null value and all columns are in float data type
print("Train Dataset Information\n")
print(train_case_df.info())
print("Test Dataset Information\n")
print(test_case_df.info())

# Get the half number of row from test dataset
test_rows = len(test_case_df.index)
split_num = int(test_rows/2)

# Insert the dataframe into data and perform splitting of test dataset into testing and validation data
train_data = train_case_df
test_data = test_case_df[:split_num]
val_data = test_case_df[split_num:]

# Show the shape size for train, test and validation dataset
print("Train dataset shape:",train_data.shape)
print("Test dataset shape:",test_data.shape)
print("Validation dataset shape:",val_data.shape)

# Get the mean and standard deviation of train dataset
train_mean = train_data.mean()
train_std = train_data.std()

# Get the normalized train, validation, and test dataset
train_df = (train_data - train_mean) / train_std
val_df = (val_data - train_mean) / train_std
test_df = (test_data - train_mean) / train_std

# Choose the column
chosen_column = 'cases_active'

# Using WindowGenerator function to perform data windowing
data_window = WindowGenerator(30,30,1,train_df,val_df,test_df,label_columns=[chosen_column])
print("Three batches graph of data windowing")
data_window.plot(plot_col=chosen_column)

# Setup MLFlow Experiment if not created yet
mlflow.set_experiment("Covid-19 Prediction")

# Create LSTM model for single layer with 32 units and compilation of the model
model_single_32_units = keras.Sequential()
model_single_32_units.add(keras.layers.LSTM(32,return_sequences=True))
model_single_32_units.add(keras.layers.Dense(1))

model_single_32_units.compile(optimizer='adam',loss='mse',metrics=['mae'])

# Create LSTM model for single layer with 64 units and compilation of the model
model_single_64_units = keras.Sequential()
model_single_64_units.add(keras.layers.LSTM(64,return_sequences=True))
model_single_64_units.add(keras.layers.Dense(1))

model_single_64_units.compile(optimizer='adam',loss='mse',metrics=['mae'])

# Create LSTM model for single layer with 128 units and compilation of the model
model_single_128_units = keras.Sequential()
model_single_128_units.add(keras.layers.LSTM(128,return_sequences=True))
model_single_128_units.add(keras.layers.Dense(1))

model_single_128_units.compile(optimizer='adam',loss='mse',metrics=['mae'])

# Create LSTM model for single layer with 256 units and compilation of the model
model_single_256_units = keras.Sequential()
model_single_256_units.add(keras.layers.LSTM(256,return_sequences=True))
model_single_256_units.add(keras.layers.Dense(1))

model_single_256_units.compile(optimizer='adam',loss='mse',metrics=['mae'])

# Set the maxinum number of epochs
MAX_EPOCHS = 100

# Run the experiment model for 32 units LSTM
print("LSTM 32 Units Model")
with mlflow.start_run(run_name="lstm_32_units") as run:
    mlflow.tensorflow.autolog()
    history_32 = model_single_32_units.fit(data_window.train,validation_data=data_window.val,epochs=MAX_EPOCHS)

# Run the experiment model for 64 units LSTM
print("LSTM 64 Units Model")
with mlflow.start_run(run_name="lstm_64_units") as run:
    mlflow.tensorflow.autolog()
    history_64 = model_single_64_units.fit(data_window.train,validation_data=data_window.val,epochs=MAX_EPOCHS)

# Run the experiment model for 128 units LSTM
print("LSTM 128 Units Model")
with mlflow.start_run(run_name="lstm_128_units") as run:
    mlflow.tensorflow.autolog()
    history_128 = model_single_128_units.fit(data_window.train,validation_data=data_window.val,epochs=MAX_EPOCHS)

# Run the experiment model for 256 units LSTM
print("LSTM 256 Units Model")
with mlflow.start_run(run_name="lstm_256_units") as run:
    mlflow.tensorflow.autolog()
    history_256 = model_single_256_units.fit(data_window.train,validation_data=data_window.val,epochs=MAX_EPOCHS)

# Model architecture of the LSTM 32 Units model
print("LSTM 32 Units Model Architecture")
model_single_32_units.summary()

# Model architecture of the LSTM 64 Units model
print("LSTM 64 Units Model Architecture")
model_single_64_units.summary()

# Model architecture of the LSTM 128 Units model
print("LSTM 128 Units Model Architecture")
model_single_128_units.summary()

# Model architecture of the LSTM 256 Units model
print("LSTM 256 Units Model Architecture")
model_single_256_units.summary()


# Load the best lstm model
model_load = mlflow.tensorflow.load_model(model_uri=f"models:/best_lstm_model/6")

# Show the prediction of the model based on data windowing test dataset
print("Predicted normalized output from the normalized test dataset")
predictions = model_load.predict(data_window.test)
predictions_squeezed = predictions.squeeze(axis=-1)  # Remove the last dimension
predictions_df = pd.DataFrame(predictions_squeezed)
print(predictions_df)

# Plot the graph for the best model
print("Predictions for the first three batches on the best model")
data_window.plot(plot_col=chosen_column,model=model_load)
# %% Show the model architecture for the best model
print("Best model architecture")
model_load.summary()

