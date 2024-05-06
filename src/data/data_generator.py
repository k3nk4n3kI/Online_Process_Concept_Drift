def next_activity(df):
    df['next activity'] = df.groupby('case:concept:name')['concept:name'].shift(-1)

    # Replace last activity of each case with 'N.A'
    last_row_indices = df.groupby('case:concept:name').tail(1).index
    df.loc[last_row_indices, 'next activity'] = 'end'

    return df