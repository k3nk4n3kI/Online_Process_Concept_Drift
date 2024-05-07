import pandas as pd
import os
from datetime import datetime
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

#Function to load an CSV or XES file into the model
def data_loader(file, columns):
    '''
    Function that loads the initial dataset into the model and standardizes the column names. Furthermore, only predefined columns are considered.
    Next the data type of the time column gets standardized and the dataframe is sorted by case number and timestamp.

    Input:
        -file:  A .csv or .xes file that contains the data
        -columns: Columns that should be included in the dataset, named by their name in the initial file

    Output:
        -df: A dataframe with standardized column names and sorted by case number and time

    '''

    if file[-4:] == ".csv":
        df = pd.read_csv(file)
    elif file[-4:] == ".xes":
        log = xes_importer.apply(file)
        df = xes_converter.apply(log, variant=xes_converter.Variants.TO_DATA_FRAME)
    else:
        print("File could not be loaded")
    
      # Rename columns
    standard_columns = ['time:timestamp', 'case:concept:name', 'concept:name', 'org:resource']
    new_column_names = {columns[i]: standard_columns[i] for i in range(len(columns))}

    df = df.rename(columns=new_column_names)
    # Include only the specified columns
    df = df[new_column_names.values()]

    # Column name timestamps
    column_name = 'time:timestamp'

    # Check if timestamps are in the right format and convert if not
    column_type = df[column_name].dtypes

    if str(column_type) != 'datetime64[ns, UTC]':
        df[column_name] = pd.to_datetime(df[column_name])

    # Sort by case:concept:name and time:timestamp
    if column_name in new_column_names.values():
        df = df.sort_values(by=['case:concept:name', column_name])
    else:
        print('Data is not sorted by case and timestamp, because the column names don\'t correspont to the standard naming scheme.')
    return df

#-------------------------------------------------------------------------------

#Function to save interim dataset
def save_event_log(df, dataset_name):
    '''
    Function to store interim datasets with uniform naming convention as a pickle file.
    Naming convention: current date + dataset name + "next_activity.pkl"

    Input:
        -df: Processed dataset, that should saved
        -dataset_name: Name of the specific dataset, which is used for the specific naming convention

    Output:
        - Saved .pkl file in the data/interim folder 

    '''

    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Define the file name
    file_name = f'{current_date}_{dataset_name}_next_activity.pkl'

    # Define the full path to the file
    file_path = os.path.join('/Users/lars/Meine Ablage/01_Universität/01_Masterarbeit/02_Programmierung/Online_Process_Concept_Drift/data/interim', file_name)

    # Save DataFrame to pickle in the 'interim' folder
    df.to_pickle(file_path)

    print(f"{file_name} has been saved.")

#-------------------------------------------------------------------------------

#Function to load most recent dataframe
def load_event_log(dataset_name):
    '''
    Loads the most current version of a certain preprocessed pickle file from the interim folder into the enviroment. 

    Input:
        -dataset_name: Name of the preprocessed dataset that should be loaded

    Output:
        -df: Preprocessed dataset

    '''

    # Define the directory path
    directory = '/Users/lars/Meine Ablage/01_Universität/01_Masterarbeit/02_Programmierung/Online_Process_Concept_Drift/data/interim'

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter only pickle files for the specified dataset
    pickle_files = [file for file in files if file.endswith('.pkl') and dataset_name in file]

    if not pickle_files:
        print(f"No pickle files found for dataset '{dataset_name}'")
        return None

    # Sort the list of pickle files by their names (assuming the names contain the date)
    pickle_files.sort(reverse=True)

    # Get the most recent pickle file
    most_recent_file = pickle_files[0]

    # Load the most recent DataFrame
    df = pd.read_pickle(os.path.join(directory, most_recent_file))

    print(f"{dataset_name} loaded")
    
    return df

#-------------------------------------------------------------------------------

#Function to delete all dataframes to reduce memory load
def delete_dataframes():
    '''
    Deletes all variables that contain a pandas dataframe, to reduce the total memory consumption.
    '''
    
    global_vars = globals()
    for var_name in list(global_vars.keys()):
        if isinstance(global_vars[var_name], pd.DataFrame):
            del global_vars[var_name]

    print("All dataset varibales are deleted")

#-------------------------------------------------------------------------------