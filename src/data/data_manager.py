import pandas as pd
import os
import shutil
import glob
from datetime import datetime
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import tensorflow as tf

#Function to load an CSV or XES file into the model
def data_loader(directory, folder, dataset, columns=None):
    '''
    Function which loads .csv and .xes files into the working enviroment. Based on the input different states of a dataset
    can be loaded e.g. raw, interim or processed.
    Afterwards, the data type of the time column gets standardized and the dataframe is sorted by case number and timestamp.

    Input:
        - file: str - A .csv or .xes file that contains the data
        - columns (optinal): list - Columns that should be included in the dataset, named by their name in the initial file

    Output:
        - df: dataframe - A dataframe with standardized column names and sorted by case number and time

    '''
    file = directory + folder + dataset
    path = directory + folder
    dataset_name = dataset

    #Checks if which kind of data is wanted (raw, interim or processed)
    if folder == "/data/raw/":
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

    elif folder == "/data/interim/":
            # Get a list of all files in the directory
        dataset_name = dataset
        files = os.listdir(path)

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
        df = pd.read_pickle(os.path.join(path, most_recent_file))

    elif folder == "/data/processed/":
        # List all folders in the save directory
        folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

        # Filter folders based on the dataset name
        filtered_folders = [folder for folder in folders if dataset in folder]

        # Sort the folders by date in descending order
        sorted_folders = sorted(filtered_folders, reverse=True)

        # Get the path of the most recent folder
        if sorted_folders:
            most_recent_folder = os.path.join(path, sorted_folders[0])
            print(f"Loading dataset from folder: '{most_recent_folder}'")

            # Load the dataset from the most recent folder
            loaded_dataset = tf.data.Dataset.load(most_recent_folder)
            return loaded_dataset
        else:
            print("No dataset found.")
            return None

    else:
        print("Path or file name are wrong")


    return df

#-------------------------------------------------------------------------------

#Function to save dataset
def save_event_log(directory, folder, df, dataset_name):
    '''
    Function which saves datasets with uniform naming convention into the specific folder.
    Naming convention: current date + dataset name + "next_activity.pkl"

    Input:
        - directory: str - path of the current working directory
        - folder: str - path to the folder file should be stored
        - df: dataframe - Processed dataset, that should saved
        - dataset_name: str - Name of the specific dataset, which is used for the specific naming convention

    Output:
        - Saved .pkl file in the data/interim folder 

    '''

    path = directory + folder

    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    if folder == "/data/interim/":

        # Define the file name
        file_name = f'{current_date}_{dataset_name}_next_activity.pkl'

        # Define the full path to the file
        file_path = os.path.join(path, file_name)

        # Save DataFrame to pickle in the 'interim' folder
        df.to_pickle(file_path)

        print(f"File saved as {file_name}")

    elif folder == "/data/processed/":

        file_name = f"{current_date}_ {dataset_name}_tensor"
        file_path = os.path.join(path, file_name)

        # If file already exists, replace it
        if os.path.exists(file_path):
            try:
                shutil.rmtree(file_path)
                print(f"Removed existing folder '{file_path}'")
            except Exception as e:
                print(f"Error removing folder '{file_path}': {e}")

        # Save the dataset to the folder
        tf.data.Dataset.save(df, file_path)
        print(f"Saved new folder '{file_path}'")


        print(f"File saved as {file_name}")

    else:
        print("Unsupported data type. Only Pandas DataFrame and TensorFlow dataset are supported.")

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