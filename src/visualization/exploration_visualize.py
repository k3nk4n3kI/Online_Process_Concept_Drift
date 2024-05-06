import pm4py

def plot_bpmn_map(df):
    # Create and display process model
    process_model = pm4py.discover_bpmn_inductive(df)
    pm4py.view_bpmn(process_model)

    return process_model

#-------------------------------------------------------------------------------

def plot_petri_net(bpmn_map):
    # Create and display petri net
    net, im, fm = pm4py.convert_to_petri_net(bpmn_map)
    pm4py.view_petri_net(net, im, fm, format='png')

    return net, im, fm

#-------------------------------------------------------------------------------


def plot_dfg(df, num_variants=5):
    # Filter dataframe for top variant
    filtered_dataframe = pm4py.filter_variants_top_k(df, num_variants, activity_key='concept:name', 
                                                     timestamp_key='time:timestamp', case_id_key='case:concept:name')
    # Create and plot directly follows graph
    dfg, start_activities, end_activities = pm4py.discover_dfg(filtered_dataframe, case_id_key='case:concept:name', 
                                                               activity_key='concept:name', timestamp_key='time:timestamp')
    pm4py.view_dfg(dfg, start_activities, end_activities)

    return dfg, start_activities, end_activities
