import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import tensorflow as tf

def outliers(dataframe, case_column):         #case_column as string
    '''
    Function for identifying and deleting outliers. 
    By calculating the Inter Quantil Range of the case length an upper and lower bound is defined.
    Cases outside the upperbound are deleted. Cases outside the lower bound are deleted if they contain only one event.

    Input:
        -dataframe: dataframe - A dataframe whose outliers should be deleted
        -case_column: str - Column name that contains the case IDs

    Output:
        -filtered_event_log: dataframe - Dataframe whose outliers are deleted

    '''

    event_counts = dataframe[case_column].value_counts()

    #Calculate quantiles
    Q3 = event_counts.quantile(0.75)
    Q1 = event_counts.quantile(0.25)

    #Calculate IQR
    IQR = Q3 - Q1

    #Calculate uper bound
    upper_bound = Q3 + 1.5 * IQR

    #Filter cases
    filtered_event_log = dataframe[dataframe[case_column].isin(event_counts[(event_counts <= upper_bound) & (event_counts >1)].index)]

    #After df has been filtered it is filtered again for its 99% quantile
    #This ensures that the final token length never exceeds 512

    event_counts_filtered = filtered_event_log[case_column].value_counts()
    Q99 = event_counts_filtered.quantile(0.99)

    # Filter cases based on the 99% quantile
    filtered_event_log = filtered_event_log[filtered_event_log[case_column].isin(event_counts_filtered[event_counts_filtered <= Q99].index)]

    return filtered_event_log

#-------------------------------------------------------------------------------

def missing_values(dataframe, case_column):     #case_column as string
    '''
    Function that identifies all cases that contain at least one missing value and deletes the whole case accordingly.
    Furthermore, spaces in the activity column are replaced with "-".

    Input:
        -dataframe: dataframe - Dataframe that should be checked for missing values
        -case_column: str - Column name that contains the case IDs

    Output:
        -cleaned_df: dataframe - A dataframe whose case don't contain any missing values

    '''
    
    case_ids_with_missing_values = dataframe[dataframe.isnull().any(axis=1)][case_column].unique()
    cleaned_df = dataframe[~dataframe[case_column].isin(case_ids_with_missing_values)]
    # Replace whitespaces with hyphens in the "concept:name" column
    cleaned_df['concept:name'] = cleaned_df['concept:name'].str.replace(' ', '-')

    return cleaned_df

#-------------------------------------------------------------------------------

def normalize_and_lowercase(df):
    '''
    First all numerical columns of a dataframe are identified. By applying the MinMaxScaler all numerical colunms
    are normalized to contain a value between 0 and 1. 
    Next all columns are transformed into a string data data type for later prefix trace creation.
    Finally all words are converted to be in lower case

    Input:
        -df: dataframe - Dataframe whose columns are normalized and transformed into string values

    Output:
        -new_df: dataframe - Dataframe that only contains lowercase and normalized string values

    '''

    # Select numerical columns for normalization
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # If there are numerical columns, normalize them
    if not numerical_cols.empty:

        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Normalize numerical columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Convert all columns to string format
    df = df.astype(str)
    
    # Convert all words to lowercase
    new_df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

    return new_df

#-------------------------------------------------------------------------------

def create_multiindex(df, case_column="case:concept:name", time_column="time:timestamp", org_column="org:resource"):
    '''
    This function takes as input a df and adds a multiindex based on the caseID and time to it.
    In a next step all columns besides of the activitiy and next_activity columns are deleted. This ensures that during the tokenization of
    the prefix traces all sequenzes stay below a token length of 512.
    At the end the df is sorted by time and case ID.

    Input:
        - df: dataframe - The dataframe that whose multiindex should be created
        - case_column (optional): str - Column name that contains the case IDs
        - time_comlumn (optional): str - Column name that contains the timestamps
        - org_column (optional): str - Column name that contains the org resource

    Output:
        - df: dataframe - Dataframe with new multiindex
        
    '''

    #Create Multiindex based on caseID and time column
    df.index = pd.MultiIndex.from_arrays(df[[case_column, time_column]].values.T, names=['caseID', 'time'])

    #Drop all columns besides of activity and next activity
    df.drop([case_column, time_column, org_column], axis=1, inplace=True)

    #Sort df by caseID and time
    df.sort_index(level=["caseID", "time"])

    return df


#-------------------------------------------------------------------------------

def generate_prefix_traces(df, next_activity_column='next activity'):
    '''
    This function generates prefix traces from a dataframe containing event data. First it sorts the data by a timestamp column,
    then groups the data by the case ID column. For each case, it iterates through the events in chronological order and creates
    prefix traces by including all events up to the current event. Traces are created regarding the early fusion approach. Each 
    prefix trace is stored along with the label of the next activity.

    Input:
        - df: dataframe - dataframe containing event data.
        - next_activity_column (optional): str - column name representing the next activity. Default is 'next activity'.

    Output:
        - prefix_traces: list - a list of tuples, where each tuple contains a prefix trace dataframe and its corresponding next activity label.

    '''

    prefix_traces = []

    # Group data by case ID (first level of MultiIndex)
    for case_id, case_data in df.groupby(level=0):
        # Iterate through events in chronological order
        for i in range(len(case_data)):
            # Define prefix length (e.g., all events up to the current event)
            prefix_length = i
            
            # Create prefix trace
            prefix_trace = case_data.iloc[:prefix_length + 1].copy()  # Include current event
            prefix_trace.drop(columns=[next_activity_column], inplace=True, errors='ignore')
            
            # Store prefix trace along with next activity label
            next_activity = None
            if prefix_length < len(case_data):  # Check if there's a next activity
                next_activity = case_data.iloc[prefix_length][next_activity_column]
            prefix_traces.append((prefix_trace, next_activity))

    return prefix_traces

#-------------------------------------------------------------------------------

def early_fusion(prefix_traces):
    '''
    This function creates a dataframe containing the sequenced prefix traces and the corresponding next activities.
    It takes as input a list of tuples containing dataframes of prefix traces and their corresponding next activities. 
    Next it iterates through each prefix trace, flattens it into a 1D sequence, and stores it along with the next activity
    in separate lists. Then, it creates a dataframe containing these sequences and next activities. Finally, all entries
    in the next activity column are converted to lowercase.

    Input:
        - prefix_traces: list - A list of tuples where each tuple contains a DataFrame representing a prefix trace and its corresponding next activity.

    Output:
        - df: dataframe - A dataframe that contains the sequenced prefix trace and its next activity.

    '''
    sequences = []
    next_activities = []

    # Iterate through prefix traces
    for prefix_trace, next_activity in prefix_traces:
        # Convert DataFrame to numpy array
        trace_array = prefix_trace.values
        
        # Transpose the array to order entries by column
        trace_array = trace_array.T
        
        # Flatten the transposed array to create a 1D sequence
        flattened_trace = np.concatenate(trace_array).tolist()
        
        sequences.append(flattened_trace)
        next_activities.append(next_activity)

    # Create a DataFrame with sequences and next activities
    df = pd.DataFrame({'Prefix_Trace': sequences, 'Next_Activity': next_activities})
    df['Prefix_Trace'] = df['Prefix_Trace'].map(lambda x: ' '.join(map(str, x)).lower())
    df["Next_Activity"] = df['Next_Activity'].str.lower()

    return df

#-------------------------------------------------------------------------------

def encoding_and_tokenizing(dataframe, prefix_column, activity_column):
    '''
    This function takes a dataframe which consists of prefix traces and next activities as inputs and creates a new dataset of tensors.
    First the prefix traces are tokenized by using the BertTokenizer. Next padding is applied to the tokenized sequences to ensure that
    they are of equale length. The length is determined by the longest tokenized sequence.
    To encode the next activities a dictionary is created which mapes the labels to a unique number. Next the dictionary is used to encode
    the next activities. 
    Finally the tokenized sequences and the encoded next activities are converted to tensors and merged to create the new tensor dataset.

    Input:
        - dataframe: dataframe - Dataframe that contains the prefix traces and the next activities
        - prefix_column: str - Column name that contains the prefix traces
        - activity_column: str - Colunm name that contains the next activities

    Output:
        - tensor-dataset: tf.data.Dataset - Dataset which contains the prefixes and next activities as tensors

    '''

    #Tokenization of prefix traces
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = dataframe[prefix_column].apply(lambda x:tokenizer.encode(x, add_special_tokens=True))
    #Calculating max sequence length for padding
    max_length = max(len(seq) for seq in tokenized_text)
    #Pad tokenized prefix traces
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, padding='post', maxlen=512)


    #Encode next activities
    unique_labels = dataframe[activity_column].unique()
    #Create label map using dictionary comprehension
    label_map = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = dataframe[activity_column].map(label_map)


    # Convert tokenized sequences and encoded labels to arrays
    labels = tf.constant(encoded_labels, dtype=tf.int32, name='label')
    inputs = tf.constant(padded_sequences, dtype=tf.int32, name='prefix_trace')

    tensor_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    return  tensor_dataset

#-------------------------------------------------------------------------------

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    '''
    Takes as input a pandas dataframe and splits it into train/val/test set.
    During the split it is considered that no trace is seperated by a split. If this would be the case, 
    the remaining events are included to the current set.

    Input:
        - df: dataframe - dataframe containing next acrivities
        - train_ratio (optional): float - relative size of train set
        - val_ratio (optinal): float - relative size of val set

    Output:
        - train_df: dataframe - containing the train data
        - val_df: dataframe - containing the val data
        - test_df: dataframe - containing the test data

    '''

    # Sort the dataframe by case ID and timestamp
    df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp'])

    # Calculate the number of cases for each split
    total_cases = df_sorted['case:concept:name'].nunique()
    train_cases = int(train_ratio * total_cases)
    val_cases = int(val_ratio * total_cases)
    test_cases = total_cases - train_cases - val_cases

    # Get unique case IDs
    unique_cases = df_sorted['case:concept:name'].unique()

    # Split the case IDs into train, validation, and test sets
    train_case_ids, remaining_case_ids = train_test_split(unique_cases, train_size=train_ratio, shuffle=False)
    val_case_ids, test_case_ids = train_test_split(remaining_case_ids, test_size=val_ratio / (1 - train_ratio), shuffle=False)

    # Select rows corresponding to train, validation, and test cases
    train_df = df_sorted[df_sorted['case:concept:name'].isin(train_case_ids)]
    remaining_df = df_sorted[~df_sorted['case:concept:name'].isin(train_case_ids)]
    val_df = remaining_df[remaining_df['case:concept:name'].isin(val_case_ids)]
    test_df = remaining_df[~remaining_df['case:concept:name'].isin(val_case_ids)]

    print("Train set shape:", train_df.shape)
    print("Validation set shape:", val_df.shape)
    print("Test set shape:", test_df.shape)

    return train_df, val_df, test_df

