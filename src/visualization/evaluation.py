import pickle
import matplotlib.pyplot as plt
import numpy as np
from transformers import LongformerTokenizer, BertTokenizer



# Function to plot the comparisons
def plot_comparison(data, dataset, metric):
    '''
    This function compares the fine-tuning results of the BERT base model with the LongBERT model. Therefore it creates 
    for each dataset and metric a plot which compares the train and validation results over all epochs. The metrics are the loss and the accuracy.

    Input:
        - data: dict - a dictionary that contains the loss and accuracy values for all datasets over all epochs
        - dataset: list - a list that contains the names of the individual datasets
        - metric: list - a list that contains the names of the metrics that should be analyzed

    Output:
        - plots: a plot for each dataset metric pair

    '''
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(data[dataset][metric]) + 1)
    
    # Plot for normal dataset
    plt.plot(epochs, data[dataset][metric], color="black" ,label=f'{dataset}')
    plt.plot(epochs, data[dataset]['val_' + metric], linestyle="--", color="black" ,label=f'{dataset} validation')
    
    if dataset != "bpic2018":
        # Plot for long dataset
        long_dataset = 'long_' + dataset
        plt.plot(epochs, data[long_dataset][metric], color="blue" ,label=f'{long_dataset}')
        plt.plot(epochs, data[long_dataset]['val_' + metric], linestyle="--", color="blue" ,label=f'{long_dataset} validation')

    if metric == "accuracy":
        plt.ylim(0.5,1)
    plt.title(f'{metric.capitalize()} Comparison for {dataset}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to calculate and print averages over epochs
def calculate_epoch_averages(data, metric):
    '''
    This function calculates the average train/val loss and accuracy for the first 5 epochs of LongBERT and BERT base.
    The average is calulated by using the mean for each metric over the individual epoch for all datasets the model was trained.
    The function differitiates between LongBERT and BERT base by adding "long_" in fornt of the metric name.

    Input:
        - data: dict - a dictionary that contains the dataset name as key and the evaluation results as another dictionary as values.
                       The dictionary with the evaluation results consists of the metric name as key and distinct epoch values as values.
        - metric: str - a string that defines for which metric the average should be calculated.
    
    Output:
        avg_values: dict -  a dictionary holind the label names as keys and the average values as a list as values.

    '''

    labels = ['train', 'val', 'long_train', 'long_val']
    avg_values = {label: [] for label in labels}

    for epoch in range(5):
        avg_values['train'].append(np.mean([data[dataset][metric][epoch] for dataset in ['helpdesk', 'bpic2012', "bpic2018"]]))
        avg_values['val'].append(np.mean([data[dataset]['val_' + metric][epoch] for dataset in ['helpdesk', 'bpic2012', "bpic2018"]]))
        avg_values['long_train'].append(np.mean([data['long_' + dataset][metric][epoch] for dataset in ['helpdesk', 'bpic2012']]))
        avg_values['long_val'].append(np.mean([data['long_' + dataset]['val_' + metric][epoch] for dataset in ['helpdesk', 'bpic2012']]))
    
    return avg_values


# Function to plot average comparisons over epochs
def plot_average_over_epochs(avg_values, metric):
    '''
    Function that creates a plot which compares the average train/val results for BERT base and LongBERT over all epochs of a certrain metric.

    Input:
        - avg_values: dict - a dictionary holind the label names as keys and the average values as a list as values.
        - metric: str - used for the title to clearify which metric is beeing shown.
    
    '''

    labels = ['train', 'val', 'long_train', 'long_val']
    epochs = range(1, 6)
    colors = {'train': 'black', 'val': 'black', 'long_train': 'red', 'long_val': 'red'}
    linestyles = {'train': '-', 'val': '--', 'long_train': '-', 'long_val': '--'}

    plt.figure(figsize=(8, 5))
    if metric == "accuracy":
        plt.ylim(0.5,1)
    for label in labels:
        plt.plot(epochs, avg_values[label], color=colors[label], linestyle=linestyles[label], label=label.replace('_', ' ').capitalize())
    
    plt.title(f'Average {metric.capitalize()} Over Epochs for Normal and Long Datasets')
    plt.xlabel('Epochs')
    plt.ylabel(f'Average {metric.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()


def tokenize_text(data, column):
    '''
    Function that tokenizes the prefix traces of an event log in a way suitable for LongBERT.

    Input:
        - data: df - a dataframe storing the prefix traces of an event log
        - column: str - name of column, which contains the prefix traces

    Output:
        - tokenized_sequences: list - inherits the tokenized sequences suitable for LongBERT

    '''

    # Load the Longformer tokenizer
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    # Apply Longformer tokenizer to the specified column
    tokenized_sequences = [tokenizer.encode(text, add_special_tokens=True) for text in data[column]]
    return tokenized_sequences

# Function to create frequency distribution
def create_freq_distribution(sequence_lengths):
    '''
    Function which calulates how often each unique sequence length appears in the tokenized prefix traces.

    Input:
        - sequence_length: list - containing the length of all tokenized sequences for a certrain event log.

    Output:
        - dictionary: dict -  a dictionary which contains the unique sequence length as key and the amount for each sequence length as value.

    '''

    unique, counts = np.unique(sequence_lengths, return_counts=True)
    return dict(zip(unique, counts))

def plot_barchart_helpdesk(freq1, freq2, label1, label2, title):
    '''
    Creates a barchart which compares the amount of each sequence length for the normal event log with the shortend one.

    Input:
        - freq1: dict - dictionary containing the frequency distribution for the short dataset.
        - freq2: dict - dictionary containing the frequency distribution for the long dataset.
        - label1: str - label for the first dataset.
        - label2: str - label for the second dataset.
        - title: str - title of the chart

    '''

    # Ensure the bins are tuples
    lengths = sorted(set(freq1.keys()).union(set(freq2.keys())), key=lambda x: x[0] if isinstance(x, tuple) else x)
    
    # Handle the case where lengths might not be tuples
    if all(isinstance(length, tuple) for length in lengths):
        x_labels = [f'{int(length[0])}-{int(length[1])}' for length in lengths]
    else:
        x_labels = [str(length) for length in lengths]

    values1 = [freq1.get(length, 0) for length in lengths]
    values2 = [freq2.get(length, 0) for length in lengths]
    
    x = np.arange(len(lengths))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, values1, width, label=label1)
    rects2 = ax.bar(x + width/2, values2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sequence Length (binned)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()

    plt.show()

def plot_barchart_helpdesk_zoom(freq1, freq2, label1, label2, title, start=26, end=36):
    '''
    Creates a barchart which zooms in on a given range of frequencies to analyse their distribution in detail.
    Plot specifically used for Helpdesk dataset.

    Input:
        - freq1: dict - dictionary containing the frequency distribution for the short dataset.
        - freq2: dict - dictionary containing the frequency distribution for the long dataset.
        - label1: str - label for the first dataset.
        - label2: str - label for the second dataset.
        - title: str - title of the chart
        - start (optinal): int - integer that defines start frequence (default=26)
        - end (optinal): int - integer that defines end frequence (default=36)

    '''

    lengths = sorted(set(freq1.keys()).union(set(freq2.keys())))
    lengths = [length for length in lengths if start <= length <= end]
    values1 = [freq1.get(length, 0) for length in lengths]
    values2 = [freq2.get(length, 0) for length in lengths]
    
    x = np.arange(len(lengths))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, values1, width, label=label1)
    rects2 = ax.bar(x + width/2, values2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.legend()

    fig.tight_layout()

    plt.show()

# Function to create binned frequency distribution
def create_binned_freq_distribution(sequence_lengths, bin_size=10):
    '''
    Function that based on a list of frequencies length creates bins of a predefined size (default = 10).

    Input:
        - sequence_lengths: list - containing the length of all tokenized sequences for a certrain event log.
        - bin_size (optional): int - defines the bin size (default = 10)
    
    Output:
        - dictionary: dict - a dictionary containing a tuple as key and the amount of sequences as value. The tuple defines the current interval of the sequence length.

    '''

    max_length = max(sequence_lengths)
    bins = np.arange(0, max_length + bin_size, bin_size)
    binned_counts = np.histogram(sequence_lengths, bins=bins)[0]
    bin_labels = [(bins[i], bins[i+1]-1) for i in range(len(bins)-1)]
    return dict(zip(bin_labels, binned_counts))

def plot_barchart_bpic2012(freq1, freq2, label1, label2, title, min_length=0):
    '''
    Creates a barchart which compares the amount of each sequence length for the normal event log with the shortend one.
    The sequence length are presented as bins.
    With the parameter min_length it is possible to zoom in on the distribution for the highest sequence lengths.

    Input:
        - freq1: dict - dictionary containing the frequency distribution for the short dataset.
        - freq2: dict - dictionary containing the frequency distribution for the long dataset.
        - label1: str - label for the first dataset.
        - label2: str - label for the second dataset.
        - title: str - title of the chart.
        - min_length (optional): int - minimum sequence length to be included in the chart. (default = 0)

    '''

    # Filter lengths based on min_length
    lengths = sorted(set(freq1.keys()).union(set(freq2.keys())))
    lengths = [length for length in lengths if length[0] >= min_length]
    values1 = [freq1.get(length, 0) for length in lengths]
    values2 = [freq2.get(length, 0) for length in lengths]
    
    x = np.arange(len(lengths))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, values1, width, label=label1)
    rects2 = ax.bar(x + width/2, values2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sequence Length Range')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l[0]}-{l[1]}" for l in lengths], rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()

    plt.show()

def plot_barchart_bpic2018(freq1, freq2, label1, label2, title):
    '''
    Creates a barchart which compares the amount of each sequence length for the normal event log with the shortend one.
    The sequence length are presented as bins.

    Input:
        - freq1: dict - dictionary containing the frequency distribution for the short dataset.
        - freq2: dict - dictionary containing the frequency distribution for the long dataset.
        - label1: str - label for the first dataset.
        - label2: str - label for the second dataset.
        - title: str - title of the chart

    '''
    
    # Sorting bins based on numeric value
    lengths = sorted(set(freq1.keys()).union(set(freq2.keys())))
    values1 = [freq1.get(length, 0) for length in lengths]
    values2 = [freq2.get(length, 0) for length in lengths]
    
    x = np.arange(len(lengths))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, values1, width, label=label1)
    rects2 = ax.bar(x + width/2, values2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sequence Length (binned)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(length[0])}-{int(length[1])}' for length in lengths], rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()

    plt.show()

def plot_barchart_bpic2018_zoom(freq1, freq2, label1, label2, title, min_length=0):
    '''
    Function enables to create a bar chart showing the distribution of sequence length between the shortend and normal event log starting from a predefined sequence length.
    Shows sequence length without bins!
    Plot specifically used for BPIC2018 dataset.

    Input:
        - freq1: dict - dictionary containing the frequency distribution for the short dataset.
        - freq2: dict - dictionary containing the frequency distribution for the long dataset.
        - label1: str - label for the first dataset.
        - label2: str - label for the second dataset.
        - title: str - title of the chart
        - min_length (optional): int - defines the starting point the binned sequence lengths are plotted (default = 0)

    '''

    # Filter lengths based on min_length
    lengths = sorted(set(freq1.keys()).union(set(freq2.keys())))
    lengths = [length for length in lengths if length >= min_length]
    values1 = [freq1.get(length, 0) for length in lengths]
    values2 = [freq2.get(length, 0) for length in lengths]
    
    x = np.arange(len(lengths))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, values1, width, label=label1)
    rects2 = ax.bar(x + width/2, values2, width, label=label2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(lengths, rotation=45, ha='right')
    ax.legend()

    fig.tight_layout()

    plt.show()

def tokenize_text_base(data, column):
    '''
    Function that tokenizes the prefix traces of an event log in a way suitable for BERT base.

    Input:
        - data: df - a dataframe storing the prefix traces of an event log
        - column: str - name of column, which contains the prefix traces

    Output:
        - tokenized_sequences: list - inherits the tokenized sequences suitable for BERT base.

    '''

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Apply BERT tokenizer to the specified column without padding and truncation
    tokenized_sequences = [tokenizer.encode(text, add_special_tokens=True) for text in data[column]]
    return tokenized_sequences