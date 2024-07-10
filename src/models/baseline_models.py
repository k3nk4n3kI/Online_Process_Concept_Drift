import random
import warnings
import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from transformers import TFAutoModel, LongformerTokenizer
from tensorflow.keras.callbacks import Callback # type: ignore
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, precision_score, recall_score


#-----------------------------------------------------------
# Static model
#-----------------------------------------------------------
# Longformer

class LongformerPreprocessorStatic:
    '''
    Preprocessor class for Longformer model using Hugging Face's transformers library.

    This class handles preprocessing of text data, mapping of activity labels to integers, and preparing 
    inputs for Longformer models.
    '''

    def __init__(self, model_name='allenai/longformer-base-4096', default_max_length=4096):
        '''
        Initializes the Longformer preprocessor with default settings.

        Input:
            - model_name: str - Name of the pre-trained Longformer model. (Default = allenai/longformer-base-4096)
            - default_max_length: int - The maximum length for tokenized sequences. (Default = 4096)
        '''

        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.max_length = default_max_length
        self.activity_to_label = {}
        self.label_to_activity = {}
        self.next_label_index = 0

    def fit_activity_labels(self, df):
        '''
        Fits the label mappings based on the unique activities in the provided DataFrame.

        Input:
            - df: df - DataFrame containing the next activity data.
        '''

        # Get unique activities
        activities = df['Next_Activity'].unique()

        # Create activity to label mapping
        self.activity_to_label = {activity: idx for idx, activity in enumerate(activities)}

        # Create label to activity mapping
        self.label_to_activity = {idx: activity for activity, idx in self.activity_to_label.items()}

        # Set the next label index
        self.next_label_index = len(activities)

    def preprocess(self, df):
        '''
        Preprocesses the text and activity labels of the provided DataFrame, preparing the inputs expected by the Longformer model.

        Input:
            - df: df - DataFrame containing prefix traces and next activity data.

        Output:
            - tuple: A tuple containing a dictionary of the preprocessed Longformer inputs and the maximum length.
        '''

        text_inputs = df['Prefix_Trace'].tolist()
        
        # Convert activity labels to integers
        labels = []
        for next_activity in df['Next_Activity']:
            if next_activity not in self.activity_to_label:
                self.activity_to_label[next_activity] = self.next_label_index
                self.label_to_activity[self.next_label_index] = next_activity
                self.next_label_index += 1
            labels.append(self.activity_to_label[next_activity])

        # Convert labels to tensors
        labels = tf.convert_to_tensor(labels)

        # Tokenize the text inputs
        encodings = self.tokenizer(text_inputs, padding='max_length', max_length=self.max_length, return_tensors='tf')

        # Check for any out-of-bounds token IDs
        input_ids = encodings['input_ids']
        vocab_size = self.tokenizer.vocab_size
        if tf.reduce_max(input_ids) >= vocab_size:
            print(f"Max token ID: {tf.reduce_max(input_ids)}")
            print(f"Vocabulary size: {vocab_size}")
            raise ValueError("Some token IDs are out of bounds of the tokenizer vocabulary size.")

        preprocessed_text = {
            'input_ids': input_ids,
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }

        return preprocessed_text, self.max_length  # Return the dynamically calculated max_length

#-----------------------------------------------------------

# Define a class to build the LongBERT model

class LongformerModelBuilderStatic:
    '''
    Builder class for creating a Longformer model for classification tasks.
    This class handles the creation and compilation of a Longformer model for the next activity prediction by
    using TensorFlow and Hugging Face's transformers.
    '''

    def __init__(self, model_name, num_classes):
        '''
        Defines which Longformer model to use and the number of unique labels

        Input:
            - model_name: str - Name of the pre-trained Longformer model from Hugging Face.
            - num_classes: int - The number of output classes for the classification task.
        '''

        self.model_name = model_name
        self.num_classes = num_classes

    def create_model(self, max_length):
        '''
        Creates and compiles the Longformer model.
        Input layers are of shape=(max_length,) to fit the output of the preprocessor class.
        Furthermore, it ensures to utilize the GPU
        '''

        # Ensure TensorFlow uses the GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # Load the pretrained Longformer
        encoder = TFAutoModel.from_pretrained(self.model_name)

        # Define the input layers input_ids and attention_mask
        input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

        # Get the encoder outputs
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the pooled_output by using the [CLS] token of the last hidden state
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]

        # Apply dropout to model
        dropout = tf.keras.layers.Dropout(rate=0.3)(pooled_output)

        # Final dense layer with softmax and L2 regularization
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype='float32',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout)

        # Create model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        
        #Compile model with AdamW optimizer and accuracy metric
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        return model

#-----------------------------------------------------------
# BERT

import tensorflow as tf
import tensorflow_hub as hub

class BERTPreprocessorStatic:
    '''
    This preprocessor class handels every aspect of transfering text data into the BERT (base) specific input.
    It tokenizes and paddes the all the data to the the  predefined max length.
    Maps the next activities (labels) to integers.
    Creates the input_word_ids, input_mask and input_types_id and transform them into tensors.
    Creates a finally dictionary concatenating all the information with the encoded labels.
    '''

    def __init__(self, default_max_length=512):
        '''
        Initializes the BERT preprocessor.

        Input:
            -default_max_length (optinal): int - The maximum length for tokenized sequences. (default = 512)
        '''
        
        # Load BERT preprocessor from TensorFlow Hub and initializes variables
        self.preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3", trainable=False)
        self.default_max_length = default_max_length
        self.max_length = default_max_length
        self.activity_to_label = {}
        self.label_to_activity = {}
        self.next_label_index = 0

    def fit_activity_labels(self, df):
        '''
        Responsible for label encoding. All unique labes are encoded in increasing order based on their appearence in the dataframe.

        Inputs:
            df: df - DataFrame containing next activity data.
        '''
        # Get unique activities
        activities = df['Next_Activity'].unique()

        # Create activity to label mapping
        self.activity_to_label = {activity: idx for idx, activity in enumerate(activities)}

        # Create label to activity mapping
        self.label_to_activity = {idx: activity for activity, idx in self.activity_to_label.items()}

        # Set the next label index
        self.next_label_index = len(activities)

    def set_max_length(self, max_length):
        '''
        Sets the maximum length for tokenized sequences, ensuring it does not exceed 512.

        Input:
            - max_length: int - The desired maximum length for tokenized sequences.
        '''

        if max_length > 512:
            raise ValueError("Maximum length cannot exceed 512")
        self.max_length = max_length

    def preprocess(self, df):
        '''
        Preprocesses the text and activity labels, converting text inputs to BERT-compatible formats, 
        and creating TensorFlow datasets for efficient batching and parallel processing.

        Input:
        - df: df - DataFrame containing the prefix traces and next activity data.

        Output:
        - dict: A dictionary containing input_word_ids, input_mask, input_types tokens as well as the encoded labels.
    
        '''

        text_inputs = df['Prefix_Trace'].tolist()

        # Encode labels
        labels = []

        for next_activity in df['Next_Activity']:
            if next_activity not in self.activity_to_label:
                self.activity_to_label[next_activity] = self.next_label_index
                self.label_to_activity[self.next_label_index] = next_activity
                self.next_label_index += 1
            labels.append(self.activity_to_label[next_activity])

        # Convert encoded activity labels into tensors
        labels = tf.convert_to_tensor(labels)

        # Create the dataset directly from text inputs
        dataset = tf.data.Dataset.from_tensor_slices(text_inputs)

        def preprocess_text(text):
            '''
            Tokenizes and pads a single text input using the BERT preprocessor.

            Input:
                - text: tensor - A tensor containing a single text input.

            Output:
                - tuple: A tuple containing padded input_word_ids, input_mask, and input_type_ids.
            '''

            # Preprocess the text input
            processed = self.preprocessor([text])

            # Pad input_word_ids, input_mask and input_type_ids
            input_word_ids = tf.keras.preprocessing.sequence.pad_sequences(processed['input_word_ids'], maxlen=self.max_length, padding='post')
            input_mask = tf.keras.preprocessing.sequence.pad_sequences(processed['input_mask'], maxlen=self.max_length, padding='post')
            input_type_ids = tf.keras.preprocessing.sequence.pad_sequences(processed['input_type_ids'], maxlen=self.max_length, padding='post')

            return input_word_ids[0], input_mask[0], input_type_ids[0]

        # Apply preprocessing in batches and in parallel
        dataset = dataset.map(lambda text: tf.py_function(func=preprocess_text, inp=[text], Tout=[tf.int32, tf.int32, tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        # Batch the dataset, prefetch for performance, and cache it
        dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE).cache()

        # Collect the batches and concatenate them into single tensors
        input_word_ids, input_mask, input_type_ids = [], [], []

        for batch in dataset:
            input_word_ids.append(batch[0])
            input_mask.append(batch[1])
            input_type_ids.append(batch[2])

        # Concatenate all input_word_ids, input_masks and input_type_ids batches into their own individual tensor
        input_word_ids = tf.concat(input_word_ids, axis=0)
        input_mask = tf.concat(input_mask, axis=0)
        input_type_ids = tf.concat(input_type_ids, axis=0)

        with tf.device('/device:GPU:0'):
            input_word_ids = tf.convert_to_tensor(input_word_ids)
            input_mask = tf.convert_to_tensor(input_mask)
            input_type_ids = tf.convert_to_tensor(input_type_ids)

        #Create final dictionary, containing the preprocessed text and labels
        preprocessed_text = {
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids,
            'labels': labels
        }

        return preprocessed_text

#-----------------------------------------------------------

class BERTModelBuilderStatic:
    '''
    Builder class for creating a BERT model for classification tasks.

    This class handles the creation and compilation of a BERT model for the next activity prediction using TensorFlow and TensorFlow Hub.
    The class takes advantage of a preprocessed BERT (base) model from TensorFlow Hub.
    '''
    def __init__(self, preprocessor_url, encoder_url, num_classes, max_length=512):
        '''
        Initializes the model builder with the preprocessor and encoder URL, number of classes, and maximum length.

        Inputs:
            - preprocessor_url: str - The URL of the pre-trained BERT preprocessor from TensorFlow Hub.
            - encoder_url: str - The URL of the pre-trained BERT encoder from TensorFlow Hub.
            - num_classes: int - The number of output classes for the classification task.
            - max_length: int - The maximum length for tokenized sequences.
        '''
        self.preprocessor_url = preprocessor_url
        self.encoder_url = encoder_url
        self.num_classes = num_classes
        self.max_length = max_length

    def create_model(self):
        '''
        Creates and compiles the BERT model.
        Input layers are in shape=(max_length,) to fit the output of the preprocessor class

        Output:
            - model: tf.keras.Model - A compiled TensorFlow BERT model.
        '''

        # Load the preprocessor and encoder from TensorFlow Hub
        preprocessor = hub.KerasLayer(self.preprocessor_url, name='preprocessing')
        encoder = hub.KerasLayer(self.encoder_url, trainable=True, name='BERT_encoder')

        # Define input layers
        input_word_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32, name='input_type_ids')

        # Create a dictionary of encoder inputs
        encoder_inputs = {
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        }

        # Pass the inputs through the BERT encoder and receive the pooled outputs
        encoder_outputs = encoder(encoder_inputs)
        pooled_output = encoder_outputs['pooled_output']

        # Apply dropout to avoid overfitting
        dropout = tf.keras.layers.Dropout(rate=0.3)(pooled_output)

        # Final dense layer with softmax activation and L2 regularization
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype='float32',
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout)

        # Create the model
        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)

        #Compile the model with AdamW optimizer and accuracy metric
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        return model
    
#-----------------------------------------------------------

# Adding custom callback class to retrieve F1 Score, Precision and Recall

class MetricsCallbackStatic(Callback):
    '''
    Custom callback for calculating, printing, and storing F1 Score, Precision, and Recall at the end of each epoch during training.
    '''

    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        # Initialize lists to store the metrics
        self.f1_scores = []
        self.precisions = []
        self.recalls = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = []
        val_true = []

        # Process each batch in validation dataset
        for batch in self.validation_data:
            x_val, y_val = batch
            # Model predictions with silent prediction
            y_pred = self.model.predict(x_val, verbose=0)
            y_pred = tf.argmax(y_pred, axis=1)

            # Accumulate predictions and true labels
            val_pred.extend(y_pred.numpy())
            val_true.extend(y_val.numpy())

        # Calculate metrics
        f1 = f1_score(val_true, val_pred, average='weighted')
        precision = precision_score(val_true, val_pred, average='weighted', zero_division=0)
        recall = recall_score(val_true, val_pred, average='weighted', zero_division=0)

        # Store metrics
        self.f1_scores.append(f1)
        self.precisions.append(precision)
        self.recalls.append(recall)

        # Print metrics
        print(f' — val_f1: {f1:.4f} — val_precision: {precision:.4f} — val_recall: {recall:.4f}')

#-----------------------------------------------------------

def ensure_one_dimensional(labels):
    '''
    This function checks if the input tensor `labels` is two-dimensional with a second dimension of size 1.
    If so, it squeezes the tensor to remove the second dimension, resulting in a one-dimensional tensor.

    Input:
        - labels: tensor - The input tensor to be checked and potentially squeezed.

    Output:
        - labels: tensor -  A one-dimensional tensor
    '''

    if len(labels.shape) == 2 and labels.shape[1] == 1:
        labels = tf.squeeze(labels, axis=-1)
    return labels

#----------------------------------------------------------------------------------------------------------------------
# Dynamic models
#-----------------------------------------------------------
# Longformer

#Create classes for metrics

class MetricsCallbackDynamic(Callback):
    '''
    Custom callback for calculating, printing, and storing F1 Score, Precision, and Recall at the end of each epoch during training.
    '''

    def __init__(self, validation_data, steps_per_epoch):
        super().__init__()
        self.validation_data = validation_data
        self.steps_per_epoch = steps_per_epoch
        # Initialize lists to store the metrics
        self.f1_scores = []
        self.precisions = []
        self.recalls = []

    def on_epoch_end(self, epoch, logs=None):
        val_pred = []
        val_true = []

        # Process each batch in validation dataset
        for step, (x_val, y_val) in enumerate(self.validation_data.take(self.steps_per_epoch)):
            y_pred = self.model.predict(x_val, verbose=0)
            y_pred = tf.argmax(y_pred, axis=1)

            # Accumulate predictions and true labels
            val_pred.extend(y_pred.numpy())
            val_true.extend(y_val.numpy())

        # Calculate metrics
        f1 = f1_score(val_true, val_pred, average='weighted')
        precision = precision_score(val_true, val_pred, average='weighted', zero_division=0)
        recall = recall_score(val_true, val_pred, average='weighted', zero_division=0)

        # Store metrics
        self.f1_scores.append(f1)
        self.precisions.append(precision)
        self.recalls.append(recall)

        # Print metrics
        print(f' — val_f1: {f1:.4f} — val_precision: {precision:.4f} — val_recall: {recall:.4f}')

        # Optionally, add these metrics to the logs dictionary
        logs['val_f1'] = f1
        logs['val_precision'] = precision
        logs['val_recall'] = recall


#-----------------------------------------------------------

# Create class for Longformer model

class LongformerModelBuilderDynamic:
    '''
    Builder class for creating a Longformer model for classification tasks.
    This class handles the creation and compilation of a Longformer model for the next activity prediction by
    using TensorFlow and Hugging Face's transformers.
    '''
    def __init__(self, model_name, num_classes):
        '''
        Defines which Longformer model to use and the number of unique labels

        Input:
            - model_name: str - Name of the pre-trained Longformer model from Hugging Face.
            - num_classes: int - The number of output classes for the classification task.
        '''
        self.model_name = model_name
        self.num_classes = num_classes

    def create_model(self, max_length):
        '''
        Creates and compiles the Longformer model.
        Input layers are in shape=(None,) to handel effect of dynamic padding on the input sequences.
        Furthermore it makes sure to utilize the GPU

        Input:
            - max_length: int - The maximum input length for the model.

        Output:
            - model: tf.keras.Model - A compiled TensorFlow Longformer model.
        '''
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # Load the pre-trained Longformer model
        encoder = TFAutoModel.from_pretrained(self.model_name)

        # Define input layers for input_ids and attention_masks
        input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
        
        #Get the encoder outputs
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the pooled output and make sure it is of type tf.float32
        pooled_output = tf.keras.layers.Lambda(lambda x: tf.cast(x.pooler_output, tf.float32))(encoder_outputs)

        # Apply dropout to avoid overfitting
        dropout = tf.keras.layers.Dropout(rate=0.3)(pooled_output)

        # Final dense layer with softmax activation and L2 regularization
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype=tf.float32)(dropout)
        
        # Create the model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        # Compile the model and use AdamW for optimization
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        return model

#-----------------------------------------------------------
# Functions for dynamic padding and uniform length batching

def preprocess_function(tokenizer, example, max_length=512):    
    '''
    Tokenizes the input text using the provided tokenizer.

    Input:
        - tokenizer: The tokenizer to use for tokenizing the input text.
        - example: dataframe - A dataframe containing the input text in the column 'Prefix_Trace'.
        - max_length: int - The maximum length for the tokenized input.

    Output:
        A dictionary containing tokenized input IDs and attention masks.
    '''

    return tokenizer(example['Prefix_Trace'], padding=False, truncation=True, max_length=max_length)

#-----------------------------------------------------------

def sort_by_length(dataset, tokenizer, max_length=1024):
    '''
    Uses the preprocess_function to tokenize the dataset, and then sorts the tokenized sequences by the length of the tokenized input.

    Input:
        - dataset: dataframe - The dataset to tokenize and sort, containing the key 'Prefix_Trace' and the key 'Next_Activity'.
        - tokenizer: The tokenizer to use for tokenizing the input text.
        - max_length: int - The maximum length for the tokenized input.

    Output:
        - combined: list - A list of tuples, each containing the tokenized input, its length, and the corresponding label, sorted by length.
    '''

    # Tokenizes the dataset and calculates the length for all in input_ids
    tokenized = [preprocess_function(tokenizer, example, max_length) for example in dataset]
    lengths = [len(tok['input_ids']) for tok in tokenized]

    # Combine tokenized inputs, lengths, and labels and sort them
    combined = list(zip(tokenized, lengths, dataset['Next_Activity']))
    combined.sort(key=lambda x: x[1])

    return combined

#-----------------------------------------------------------

def create_buckets_and_batches(sorted_data, batch_size, data_collator):
    '''
    Creates batches of data from the sorted dataset, ensuring that each batch has similar length inputs.

    Input:
        - sorted_data: A list of tuples, each containing the tokenized input, its length, and the corresponding label, sorted by length.
        - batch_size: int - The number of samples per batch.
        - data_collator: The data collator to use for batching the inputs.

    Output:
        A tf.data.Dataset generator that yields batches of tokenized inputs and labels.
    '''

    def gen():
        while sorted_data:

            # Randomly select a starting index for the batch, select the batch and finally delete the selected sequences from the sorted_data list
            idx = random.randint(0, max(0, len(sorted_data) - batch_size))
            batch = sorted_data[idx:idx + batch_size]
            del sorted_data[idx:idx + batch_size]
            
            # Extract tokenized inputs and labels from the batch
            tokenized_batch = [item[0] for item in batch]
            labels = [item[2] for item in batch]
            
            # Create input dictionary 
            batch_inputs = {'input_ids': [tok['input_ids'] for tok in tokenized_batch],
                            'attention_mask': [tok['attention_mask'] for tok in tokenized_batch]}
            
            # Batch the inputs and finally yield and convert the inputs and labels to tensors
            batch_inputs = data_collator(batch_inputs)
            yield dict(batch_inputs), tf.convert_to_tensor(labels)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {'input_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
             'attention_mask': tf.TensorSpec(shape=(None, None), dtype=tf.int32)},
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

#-----------------------------------------------------------
# BERT (base) model

class BERTModelBuilderDynamic:
    '''
    Builder class for creating a BERT (base) model for classification tasks.
    This class handles the creation and compilation of a BERT (base) model for the next activity prediction by
    using TensorFlow and Hugging Face's transformers.
    '''

    def __init__(self, model_name, num_classes):
        '''
        Defines which BERT (base) model to use and the number of unique labels

        Input:
            - model_name: str - Name of the pre-trained BERT (base) model from Hugging Face.
            - num_classes: int - Number of unique labels for the classification task.
        '''

        self.model_name = model_name
        self.num_classes = num_classes

    def create_model(self):
        '''
        Creates and compiles the BERT base model.
        Input layers are in shape=(None,) to handel the effect of dynamic padding on the input sequences.
        Furthermore it makes sure to utilize the GPU

        Input:
            - max_length: int - The maximum input length for the model.

        Output:
            - model: tf.keras.Model - A compiled TensorFlow Longformer model.
        '''
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # Load the pretrained BERT model
        encoder = TFAutoModel.from_pretrained(self.model_name)

        # Input layer for input_ids and attention_masks
        input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

        # Get encoder outputs
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get the pooled output and make sure it is of type tf.float32
        pooled_output = tf.keras.layers.Lambda(lambda x: tf.cast(x.pooler_output, tf.float32))(encoder_outputs)

        # Apply dropout
        dropout = tf.keras.layers.Dropout(rate=0.1)(pooled_output)

        # Final dense layer for classification with softmax activation function and L2 regularization
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype=tf.float32,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.01))(dropout)
        
        # Create model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        # Compile model with AdamW as optimzer
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        return model

#-----------------------------------------------------------

def create_buckets_and_batches_bert(sorted_data, batch_size, data_collator):
    '''
    Creates batches of data from the sorted dataset for the BERT model. 
    Ensures that each batch has similar length inputs while shuffling the data at the start of each epoch.

    Input:
        - sorted_data: A list of tuples, each containing the tokenized input, its length, and the corresponding label, sorted by length.
        - batch_size: int - The number of samples per batch.
        - data_collator: The data collator to use for batching the inputs.

    Output:
        A tf.data.Dataset generator that yields batches of tokenized inputs and labels.
    '''

    def gen():
        while True:

            # Shuffle data at the start of each epoch
            random.shuffle(sorted_data)  

            # Iterate over the dataset and select batch
            for i in range(0, len(sorted_data), batch_size):
                batch = sorted_data[i:i + batch_size]
                
                # Skip empty batches
                if len(batch) == 0:
                    continue  
                
                # Extract tokenized inputs and labels from the batch
                tokenized_batch = [item[0] for item in batch]
                labels = [item[2] for item in batch]
                
                # Create input dictionaries
                batch_inputs = {'input_ids': [tok['input_ids'] for tok in tokenized_batch],
                                'attention_mask': [tok['attention_mask'] for tok in tokenized_batch]}
                
                # Batch the inputs and yiel the batches and labels as tensors
                batch_inputs = data_collator(batch_inputs)
                yield dict(batch_inputs), tf.convert_to_tensor(labels)
    
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {'input_ids': tf.TensorSpec(shape=(None, None), dtype=tf.int32),
             'attention_mask': tf.TensorSpec(shape=(None, None), dtype=tf.int32)},
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

