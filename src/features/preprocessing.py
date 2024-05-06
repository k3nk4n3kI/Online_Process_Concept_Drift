import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def outliers(dataframe, case_column):         #case_column as string
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

    return filtered_event_log

#-------------------------------------------------------------------------------

def missing_values(dataframe, case_column):     #case_column as string
    case_ids_with_missing_values = dataframe[dataframe.isnull().any(axis=1)][case_column].unique()
    cleaned_df = dataframe[~dataframe[case_column].isin(case_ids_with_missing_values)]
    # Replace whitespaces with hyphens in the "concept:name" column
    cleaned_df['concept:name'] = cleaned_df['concept:name'].str.replace(' ', '-')

    return cleaned_df

#-------------------------------------------------------------------------------

def normalize_and_lowercase(df):
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
    new_df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    return new_df

#-------------------------------------------------------------------------------

def generate_prefix_traces(df, sort_column, group_column, next_activity_column='next activity'):
    # Sort data by sort_column
    sorted_data = df.sort_values(by=sort_column)

    prefix_traces = []

    # Group data by group_column
    for case_id, case_data in sorted_data.groupby(group_column):
        # Iterate through events in chronological order
        for i in range(len(case_data)):
            # Define prefix length (e.g., all events up to the current event)
            prefix_length = i
            
            # Create prefix trace
            prefix_trace = case_data.iloc[:prefix_length + 1].copy()  # Include current event
            
            # Store prefix trace along with next activity label
            next_activity = None
            if prefix_length < len(case_data) - 1:  # Check if there's a next activity
                next_activity = case_data.iloc[prefix_length][next_activity_column]
            prefix_traces.append((prefix_trace, next_activity))

    return prefix_traces

#-------------------------------------------------------------------------------

def early_fusion(prefix_traces):
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