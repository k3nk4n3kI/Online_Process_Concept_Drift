def next_activity(df):
    '''
    Creates the next activity for each case by creating a new column "next activity" and shifting the activity columnn by -1.
    The last case of each trace receives the value "end".
    Furthermore the case:concept:name column is repositioned to be the first column.

    Input:
        -df: The current preprocessed dataframe

    Output:
        -df: A new dataframe that contains the next activity for each case.

    '''

    df['next activity'] = df.groupby('case:concept:name')['concept:name'].shift(-1)

    # Replace last activity of each case with 'N.A'
    last_row_indices = df.groupby('case:concept:name').tail(1).index
    df.loc[last_row_indices, 'next activity'] = 'end'
    case_column = df.pop('case:concept:name')

    # Insert the column at the first position
    df.insert(0, 'case:concept:name', case_column)

    return df