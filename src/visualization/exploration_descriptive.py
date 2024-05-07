from statistics import mean, median
import pandas as pd
import numpy as np
from scipy.stats import skew


def descriptive_analytics(df):
    '''
    This function calculates all the descriptive statistics needed for analysing all datasets.
    First it calulates the number of cases, activities and events as well as the amount of unique process instances.
    Furthermore the min/avg/med/max amount of of events per trace are determined. 
    The same is done for the amount of unique activities per trace.
    Next sparsity, variation and repetetivness are calculated.
    By calculating individual the trace lengths it is possible to determine their 99.99%,99%, 95%, 75%, 50%, 25% quantile.
    The trace length is also used to determine the skewness of a dataset.
    The amount of categorical and numerical values per dataset is also determined.
    Finally the dataset is ordered by time to calculate the min/avg/med/max of time betweeen two activities and two events.

    At the end all those results are printed.
    '''

    # Calculate the unique number of Instances, Activities and the amount of Events
    unique_instances = df["case:concept:name"].nunique()
    unique_activities = df["concept:name"].nunique()
    amount_events = df["case:concept:name"].count()

    df_sorted = df.sort_values(by=['case:concept:name', 'time:timestamp'])

    # Group the activities by CaseID
    grouped_activities_df = df_sorted.groupby('case:concept:name')['concept:name'].apply(list)

    # Function to get unique process variants for a case
    def get_unique_variants(activities):
        return tuple(activities)

    # Get unique process variants
    unique_variants_df = set(grouped_activities_df.apply(get_unique_variants))

    # Number of unique process variants
    df_variants = len(unique_variants_df)

    trace_length_df = list()

    for i in grouped_activities_df:
        trace_length_df.append(len(i))
    
    df_activities_per_instance = list()

    for i in grouped_activities_df:
        df_activities_per_instance.append(len(set(i)))

    df_sparsity = unique_activities/unique_instances
    df_variation = df_variants/unique_instances
    df_repetitivness = mean(trace_length_df)/mean(df_activities_per_instance)

    trace_length_df_array = np.array(trace_length_df)

    num_numerical_df = df.select_dtypes(include=['number']).shape[1]
    num_categorical_df = df.select_dtypes(include=['object']).shape[1]

    #Convert time column to a datetime datatype
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

    #sort dataframe and calculate the duration of each case
    df = df.sort_values(by=["case:concept:name", 'time:timestamp'])
    df['Event_Duration'] = df.groupby('case:concept:name')["time:timestamp"].diff()
    df["Case_Duration"] = df.groupby("case:concept:name")['time:timestamp'].transform(lambda x: x.max()-x.min())
    df = df.dropna(subset=["Event_Duration"])

    df["Event_Duration_Minutes"] = df["Event_Duration"] / pd.Timedelta(days=1)
    df["Case_Duration_Minutes"] = df["Case_Duration"] / pd.Timedelta(days=1)
    event_stats_df = df['Event_Duration_Minutes'].agg(['min', 'mean', 'max', 'median'])

    case_stats_df = df['Case_Duration_Minutes'].agg(['min', 'mean', 'max', 'median'])

    #Print results

    print("Amount of Instances: {}".format(unique_instances))
    print("Amount of Activities: {}".format(unique_activities))
    print("Amount of Events: {}".format(amount_events))
    print("Number of unique process variants: {}".format(df_variants))
    print("\nMinimum of Events per Instance: {}".format(min(trace_length_df)))
    print("Average of Events per Instance: {}".format(mean(trace_length_df)))
    print("Median of Events per Instance: {}".format(median(trace_length_df)))
    print("Maximum of Events per Instance: {}".format(max(trace_length_df)))
    print("\nMinimum of Activities per Instance: {}".format(min(df_activities_per_instance)))
    print("Average of Activities per Instance: {}".format(mean(df_activities_per_instance)))
    print("Median of Activities per Instance: {}".format(median(df_activities_per_instance)))
    print("Maximum of Activities per Instance: {}".format(max(df_activities_per_instance)))
    print("\nSparsity: {}".format(df_sparsity))
    print("Variation: {}".format(df_variation))
    print("Repetitivness: {}".format(df_repetitivness))
    print("\n99.99% quantile: {}".format(np.quantile(trace_length_df_array, 0.9999)))
    print("99% quantile: {}".format(np.quantile(trace_length_df_array, 0.99)))
    print("95% quantile: {}".format(np.quantile(trace_length_df_array, 0.95)))
    print("75% quantile: {}".format(np.quantile(trace_length_df_array, 0.75)))
    print("50% quantile: {}".format(np.quantile(trace_length_df_array, 0.5)))
    print("25% quantile: {}".format(np.quantile(trace_length_df_array, 0.25)))
    print("Skewness: {}".format(skew(trace_length_df_array, axis=0, bias=True)))
    print("\nNumber of numerical attributes:", num_numerical_df)
    print("Number of categorical attributes:", num_categorical_df)
    print("\nEvent Statistics:")
    print("Minimum Event Duration:", round(event_stats_df['min'], 2), "days")
    print("Average Event Duration:", round(event_stats_df['mean'], 2), "days")
    print("Maximum Event Duration:", round(event_stats_df['max'], 2), "days")
    print("Median Event Duration:", round(event_stats_df['median'], 2), "days")

    print("\nCase Statistics:")
    print("Minimum Case Duration:", round(case_stats_df['min'], 2), "days")
    print("Average Case Duration:", round(case_stats_df['mean'], 2), "days")
    print("Maximum Case Duration:", round(case_stats_df['max'], 2), "days")
    print("Median Case Duration:", round(case_stats_df['median'], 2), "days")
    