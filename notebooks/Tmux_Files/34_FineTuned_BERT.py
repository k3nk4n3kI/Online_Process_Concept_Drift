# File to fit the BERT model with the tuned parameters for all Datasets

import sys
from transformers import TFAutoModel
import random
import pandas as pd
import tensorflow as tf
import tensorflow_models as tfm
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed
from tensorflow.keras.callbacks import Callback # type: ignore
from datasets import Dataset
import os
import pickle
import time
from sklearn.metrics import f1_score, precision_score, recall_score


# Append the directory containing the src folder to sys.path
sys.path.append('/Users/lars/Documents/test/')


print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available: ", len(gpus))
if gpus:
    for gpu in gpus:
        print("GPU:", gpu)
else:
    print("No GPU available")

# Verify GPU utilization with a simple computation
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    start_time = time.time()
    c = tf.matmul(a, b)
    print("GPU computation time: ", time.time() - start_time, "seconds")


#Path variables for datasets
directory = "/Users/lars/Documents/Uni/Masterarbeit/Online_Process_Concept_Drift"
path_raw = "/data/raw/"
path_interim = "/data/interim/"
path_processed = "/data/processed/"

# Set seed for reproducability
tf.random.set_seed(1234)

with open('/home/lars.gsaenger/test/data/processed/2024-06-02_Long_BPIC2018_train_next_activity.pkl', 'rb') as file:
    train_tensor = pd.read_pickle(file)

with open('/home/lars.gsaenger/test/data/processed/2024-06-02_Long_BPIC2018_val_next_activity.pkl', 'rb') as file:
    val_tensor = pd.read_pickle(file)

with open('/home/lars.gsaenger/test/data/processed/2024-06-02_Long_BPIC2018_test_next_activity.pkl', 'rb') as file:
    test_tensor = pd.read_pickle(file)

print(train_tensor)

# Define functions for dynamic padding

def preprocess_function(tokenizer, example, max_length=512):
    return tokenizer(example['Prefix_Trace'], padding=False, truncation=True, max_length=max_length)

#-----------------------------------------------------------

def sort_by_length(dataset, tokenizer, max_length=1024):

    # Tokenizes the dataset and calculates the length for all in input_ids
    tokenized = [preprocess_function(tokenizer, example, max_length) for example in dataset]
    lengths = [len(tok['input_ids']) for tok in tokenized]

    # Combine tokenized inputs, lengths, and labels and sort them
    combined = list(zip(tokenized, lengths, dataset['Next_Activity']))
    combined.sort(key=lambda x: x[1])

    return combined

#-----------------------------------------------------------

def create_buckets_and_batches_bert(sorted_data, batch_size, data_collator):
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

print("Functions are loaded")


# Set up model

# Set the environment variable for GPU memory management
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable mixed precision for better performance and reduced memory usage
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Clear any existing GPU memory state
tf.keras.backend.clear_session()

# Reduce TensorFlow logging verbosity
tf.get_logger().setLevel('ERROR')

# Set parameters
epochs = 5
batch_size = 16
num_classes = 42
max_length = 242

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Dynamic padding and uniform length batching

# Assuming train_tensor and val_tensor are pandas dataframes
train_tensor['Prefix_Trace'] = train_tensor['Prefix_Trace'].astype(str)
val_tensor['Prefix_Trace'] = val_tensor['Prefix_Trace'].astype(str)
test_tensor['Prefix_Trace'] = test_tensor['Prefix_Trace'].astype(str)

# Convert labels to integers
label_map_train = {label: idx for idx, label in enumerate(train_tensor['Next_Activity'].unique())}
label_map_val = {label: idx for idx, label in enumerate(val_tensor['Next_Activity'].unique())}
label_map_test = {label: idx for idx, label in enumerate(test_tensor['Next_Activity'].unique())}
train_tensor['Next_Activity'] = train_tensor['Next_Activity'].map(label_map_train).astype(int)
val_tensor['Next_Activity'] = val_tensor['Next_Activity'].map(label_map_train).astype(int)
test_tensor['Next_Activity'] = test_tensor['Next_Activity'].map(label_map_train).astype(int)

# Convert to Hugging Face datasets
train_data = Dataset.from_pandas(train_tensor)
val_data = Dataset.from_pandas(val_tensor)
test_data = Dataset.from_pandas(test_tensor)

# Sort the data by length
sorted_train_data = sort_by_length(train_data, tokenizer, max_length)
sorted_val_data = sort_by_length(val_data, tokenizer, max_length)
sorted_test_data = sort_by_length(test_data, tokenizer, max_length)

# Initialize data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# Create TensorFlow datasets and ensure they repeat
tf_train_dataset = create_buckets_and_batches_bert(sorted_train_data, batch_size, data_collator)
tf_val_dataset = create_buckets_and_batches_bert(sorted_val_data, batch_size, data_collator)
tf_test_dataset = create_buckets_and_batches_bert(sorted_test_data, batch_size, data_collator)

# Prefetch datasets
tf_train_dataset = tf_train_dataset.prefetch(tf.data.AUTOTUNE)
tf_val_dataset = tf_val_dataset.prefetch(tf.data.AUTOTUNE)
tf_test_dataset = tf_test_dataset.prefetch(tf.data.AUTOTUNE)

# Calculate steps per epoch based on the length of the dataset
train_steps_per_epoch = len(sorted_train_data) // batch_size
val_steps_per_epoch = len(sorted_val_data) // batch_size
test_steps_per_epoch = len(sorted_test_data) // batch_size


# Debugging statements to check the sizes and steps
print(f"Number of training samples: {len(sorted_train_data)}")
print(f"Number of validation samples: {len(sorted_val_data)}")
print(f"Number of test samples: {len(sorted_test_data)}")
print(f"Steps per epoch (train): {train_steps_per_epoch}")
print(f"Steps per epoch (val): {val_steps_per_epoch}")
print(f"Steps per epoch (test): {test_steps_per_epoch}")


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


class BERTModelBuilderDynamic():
    def __init__(self, model_name, num_classes, batch_size, epochs):

        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs

    def create_model(self):

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

        # Final dense layer for classification with softmax activation function
        output = tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype=tf.float32)(dropout)
        
        # Create model
        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

        steps_per_epoch = int(len(train_tensor['Prefix_Trace'])/self.batch_size)
        num_train_steps = steps_per_epoch * self.epochs
        warmup_steps = int(0.1*num_train_steps)

        linear_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-5, decay_steps=num_train_steps, decay_rate=0.01)
        optimizer = tfm.optimization.lr_schedule.LinearWarmup(warmup_learning_rate=0, after_warmup_lr_sched=linear_decay, warmup_steps=warmup_steps)

        # Compile model with AdamW as optimzer
        model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=optimizer),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        return model

# Build and compile the model
model_builder = BERTModelBuilderDynamic(model_name='bert-base-uncased', num_classes=num_classes, batch_size=batch_size, epochs=epochs)
model = model_builder.create_model()
model.summary()

# Set callbacks
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
metrics_callback = MetricsCallbackDynamic(validation_data=tf_val_dataset, steps_per_epoch=val_steps_per_epoch)

start_time = time.time()

# Train the model
history_helpdesk = model.fit(
    tf_train_dataset,
    epochs=epochs,  # Increase the number of epochs if necessary
    validation_data=tf_val_dataset,
    steps_per_epoch=train_steps_per_epoch,
    validation_steps=val_steps_per_epoch,
    callbacks=[metrics_callback, early_stopping_callback]
)

end_time = time.time()

print(f"BERT (base) training time: {end_time - start_time} seconds")

print('Save model history')

# Save model weights

model_save_path = '/home/lars.gsaenger/test/models/Weights_BPIC2018_Tuned/Weights_BPIC2018_Tuned'

model.save_weights(model_save_path)

# Directories to save the model and history
history_save_dir = '/home/lars.gsaenger/test/models/models_pretrained/histories'

# Define the model and history file paths
history_save_path = os.path.join(history_save_dir, 'tuned_dynamic_bert_bpic2018_history.pkl')

# Save the history object returned by model.fit()
with open(history_save_path, 'wb') as f:
    pickle.dump(history_helpdesk.history, f)

print(f"History saved to {history_save_path}")
