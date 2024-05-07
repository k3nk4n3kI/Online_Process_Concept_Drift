import pm4py

def plot_bpmn_map(df):
    '''
    Function takes a dataframe as imput and calculates it bpmn map

    Input:
        -df: dataframe - dataframe that contains event data

    Output:
        -process_model: visual - a bpmn map of the dataset

    '''
    # Create and display process model
    process_model = pm4py.discover_bpmn_inductive(df)
    pm4py.view_bpmn(process_model)

    return process_model

#-------------------------------------------------------------------------------

def plot_petri_net(bpmn_map):
    '''
    Uses a bpmn map as input and creates a petri net visualization based on it

    Input:
        - bpmn_map: A bpmn map
    
    Output:
        - net: visual - Petri net visualization
        - initial_marking: object - initial marking
        - final_marking: obkect - final marking

    '''
    # Create and display petri net
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_map)
    pm4py.view_petri_net(net, initial_marking, final_marking, format='png')

    return net, initial_marking, final_marking

#-------------------------------------------------------------------------------


def plot_dfg(df, num_variants=5):
    '''
    Creates and displays a direct follows graph based on the top n variants of a dataframe containing event logs.

    Input:
        - df: dataframe - dataframe containing event data
        - num_variants (optional): int - number of variants that should be considered in dfg
    
    Output:
        - dfg: visual - direct follow graph
        - start_activities: object - start activity
        - end_activities: object - end activity
        
    '''
    # Filter dataframe for top variant
    filtered_dataframe = pm4py.filter_variants_top_k(df, num_variants, activity_key='concept:name', 
                                                     timestamp_key='time:timestamp', case_id_key='case:concept:name')
    # Create and plot directly follows graph
    dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_dataframe, case_id_key='case:concept:name', 
                                                               activity_key='concept:name', timestamp_key='time:timestamp')
    pm4py.view_dfg(dfg, start_activities, end_activities)

    return dfg, start_activities, end_activities
