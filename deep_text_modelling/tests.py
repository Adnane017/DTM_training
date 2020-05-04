### Import necessary packages
import os
import gzip
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.activations import relu, elu
from keras.losses import binary_crossentropy
from keras import metrics
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import talos as ta

from pyndl.preprocess import filter_event_file
from pyndl.count import cues_outcomes
from pyndl.ndl import ndl
from pyndl.activation import activation

### Set working directory
TOP = '/media/adnane/HDD drive/Adnane/PostDoc_ooominds/Programming/Deep_text_modelling_package_repo/'
# TOP = '/media/sf_PostDoc_ooominds/Programming/Deep_text_modelling_package_repo/'
WD = TOP + 'package'
os.chdir(WD)

import deep_text_modelling.preprocessing as pr
import deep_text_modelling.modelling as md
import deep_text_modelling.evaluation as ev

import imp
imp.reload(pr)
imp.reload(md)
imp.reload(ev)

### Define file paths
NAMES_FULL_CSV = TOP + "illustrative_examples/names/Data/Names_full.csv"
NAMES_FULL_GZ = TOP + "illustrative_examples/names/Data/Names_full.gz"
NAMES_FULL_CSV2 = TOP + "illustrative_examples/names/Data/Names_full2.csv"
NAMES_FULL_EPOCHS_GZ = TOP + "illustrative_examples/names/Data/Names_full_epochs.gz"
NAMES_FULL_EPOCHS_CSV = TOP + "illustrative_examples/names/Data/Names_full_epochs.csv"
NAMES_TRAIN_CSV = TOP + "illustrative_examples/names/Data/Names_train.csv"
NAMES_TRAIN_GZ = TOP + "illustrative_examples/names/Data/Names_train.gz"
NAMES_VALID_CSV = TOP + "illustrative_examples/names/Data/Names_valid.csv"
NAMES_VALID_GZ = TOP + "illustrative_examples/names/Data/Names_valid.gz"
NAMES_TEST_CSV = TOP + "illustrative_examples/names/Data/Names_test.csv"
NAMES_TEST_GZ = TOP + "illustrative_examples/names/Data/Names_test.gz"
CUE_INDEX = TOP + "illustrative_examples/names/Data/Cue_index.csv"
OUTCOME_INDEX = TOP + "illustrative_examples/names/Data/Outcome_index.csv"
TEMP_DIR = TOP + "illustrative_examples/names/Data/"

### Params
N_outcomes = 2  # number of most frequent outcomes to keep 
N_cues = 26  # number of cues to keep (all alphabet letters)

#######################################
# Conversion between csv and gz formats
#######################################

### Conversion to gz
pr.csv_to_gz(csv_infile = NAMES_FULL_CSV, gz_outfile = NAMES_FULL_GZ)
# Check the file
with gzip.open(NAMES_FULL_GZ, 'rt', encoding='utf-8') as f:   
    for x in range(3):
        print(next(f))
# Train and validation sets
csv_to_gz(csv_infile = NAMES_TRAIN_CSV, gz_outfile = NAMES_TRAIN_GZ)
csv_to_gz(csv_infile = NAMES_VALID_CSV, gz_outfile = NAMES_VALID_GZ)
# Check the file
with gzip.open(NAMES_TRAIN_GZ, 'rt', encoding='utf-8') as f:   
    for x in range(3):
        print(next(f))

### Conversion to csv
pr.gz_to_csv(gz_infile = NAMES_FULL_GZ, csv_outfile = NAMES_FULL_CSV2)
# Check the file
names_full = pd.read_csv(NAMES_FULL_CSV2)
print(f'Number of examples: {len(names_full)}')
names_full.head(5)

#################################################
# Export a dataframe into a tab seperated gz file
#################################################

NAMES_TRAIN_CSV = TOP + "illustrative_examples/names/Data/Names_train.csv"
NAMES_TRAIN_FROM_DF_GZ = TOP + "illustrative_examples/names/Data/Names_train_from_df.gz"
names_train = pd.read_csv(NAMES_TRAIN_CSV, sep=',', na_filter = False)

pr.df_to_gz(data = names_train, gz_outfile = NAMES_TRAIN_FROM_DF_GZ)
# Check the file
with gzip.open(NAMES_TRAIN_FROM_DF_GZ, 'rt', encoding='utf-8') as f:   
    for x in range(3):
        print(next(f))


################
# Create epochs
################

pr.create_epochs_textfile(infile_path = NAMES_FULL_GZ, outfile_path = NAMES_FULL_EPOCHS_GZ, epoch = 10, shuffle = False)

# Check the file 
epoch_file = pr.IndexedFile(NAMES_FULL_EPOCHS_GZ, 'gz')
len(epoch_file) # number of lines
pr.gz_to_csv(gz_infile = NAMES_FULL_EPOCHS_GZ, csv_outfile = NAMES_FULL_EPOCHS_CSV)

############
# Train NDL
############

cue_to_index = pr.import_index_system(CUE_INDEX, N_tokens = N_cues)
outcome_to_index = pr.import_index_system(OUTCOME_INDEX)

# Load the train and validation datasets
names_train = pd.read_csv(NAMES_TRAIN_CSV, sep=',', na_filter = False)
names_valid = pd.read_csv(NAMES_VALID_CSV, sep=',', na_filter = False)
names_test = pd.read_csv(NAMES_VALID_CSV, sep=',', na_filter = False)

# Model fitting
p = {'epochs': 2, 'lr': 0.001}
NDL_history_dict, NDL_model = md.train_NDL(data_train = names_train, 
                                            data_valid = names_valid, 
                                            cue_index = cue_to_index, 
                                            outcome_index = outcome_to_index, 
                                            temp_dir = TEMP_DIR,
                                            chunksize = 5426,
                                            shuffle = False, 
                                            num_threads = 16, 
                                            verbose = 1,
                                            metrics = ['accuracy', 'precision', 'recall', 'f1score'], 
                                            metric_average = 'macro',
                                            params = p)

# Save the weights and training history
MODEL_PATH = TOP + 'illustrative_examples/names/Results/NDL_names.h5'
HISTORY_PATH = TOP + 'illustrative_examples/names/Results/NDL_history_dict_names'
md.export_model(model = NDL_model, path = MODEL_PATH)

#FNN_model.save(MODEL_PATH)  # creates a HDF5 file 
md.export_history(history_dict = NDL_history_dict, path = HISTORY_PATH)
del NDL_model, NDL_history_dict  # deletes the existing model and history dictionary

# Load the model and training history
NDL_model = md.import_model(MODEL_PATH)
NDL_history_dict = md.import_history(path = HISTORY_PATH)

#################
# Evaluate NDL
#################

cue_seq = 'y_o_u_s_s_e_f'
model = NDL_model

from deep_text_modelling.evaluation import activations_to_proba

### Extract the cue tokens 
cues = cue_seq.split('_')

model.weights.loc[{'outcomes': 'f', 'cues': cues}].values
model.weights.loc[{'cues': cues}].values.sum(axis=1)

### Compute the activations for all outcomes based on the cues that appear in the weight matrix
activations = model.weights[:, cues].sum(axis=1)

### Convert to the activations to probabilities
proba_pred = activations_to_proba(activations, T = 1)

return proba_pred


#################
# Grid search NDL
#################

TUNING_OUTPUT_FILE = TOP + 'illustrative_examples/names/Results/grid_search_NDL_names.csv'
p = {'epochs': [1,2,3], 
     'lr': [0.001, 0.002]}
md.grid_search_NDL(data_train = names_train, 
                   data_valid = names_valid, 
                   cue_index = cue_to_index, 
                   outcome_index = outcome_to_index, 
                   temp_dir = TEMP_DIR,
                   chunksize = 5426,
                   params = p, 
                   prop_grid = 0.5, 
                   tuning_output_file = TUNING_OUTPUT_FILE, 
                   shuffle = False, 
                   num_threads = 16, 
                   verbose = 1,
                   metrics = ['accuracy', 'precision', 'recall', 'f1score'], 
                   metric_average = 'macro')

############
# Train FNN
############

### Hyperparameters to use
p = {'epochs': 3, # number of iterations on the full set 
    'batch_size': 16, 
    'hidden_layers': 2, # number of hidden layers 
    'hidden_neuron':64, # number of neurons in the input layer 
    'lr': 0.0001, # learning rate       
    'dropout': 0.2, 
    'optimizer': Adam, 
    'losses': binary_crossentropy, 
    'activation': relu, 
    'last_activation': 'sigmoid'}

# Model fitting
FNN_out, FNN_model = md.train_FNN(data_train = names_train, 
                                  data_valid = names_valid, 
                                  cue_index = cue_to_index, 
                                  outcome_index = outcome_to_index, 
                                  generator = md.generator_df_FNN,
                                  verbose = 2,
                                  params = p)

# Save the weights and training history
MODEL_PATH = TOP + 'illustrative_examples/names/Results/FNN_names_test.h5'
HISTORY_PATH = TOP + 'illustrative_examples/names/Results/FNN_history_dict_names_test'
md.export_model(model = FNN_model, path = MODEL_PATH)
md.export_history(history_dict = FNN_out.history, path = HISTORY_PATH)
del FNN_model, FNN_out  # deletes the existing model and history dictionary

# Load the model and training history
FNN_model = md.import_model(path = MODEL_PATH, custom_measures = {'precision': ev.precision, 'recall': ev.recall, 'f1score': ev.f1score})
FNN_history_dict = md.import_history(path = HISTORY_PATH)

##################

data_train = NAMES_TRAIN_GZ
data_valid = NAMES_VALID_GZ 
cue_index = cue_to_index
outcome_index = outcome_to_index 
temp_dir = TEMP_DIR
data_type = 'gz'
shuffle = False 
num_threads = 6
verbose = 0
metrics = ['accuracy', 'precision', 'recall', 'f1score']
params = {'epochs': 2, 'lr': 0.0001}
chunksize = 5426


#from deep_text_modelling.evaluation import activations_to_predictions

### Paths of the files
events_train_path = data_train
events_valid_path = data_valid
filtered_events_train_path = os.path.join(temp_dir, 'filtered_events_train.gz')  
filtered_events_valid_path = os.path.join(temp_dir, 'filtered_events_valid.gz')  

with gzip.open(events_train_path, 'rt', encoding='utf-8') as f:   
    for x in range(3):
        print(next(f))

### Filter the event files by retaining only the cues and outcomes that are in the index system (most frequent tokens)
cues_to_keep = [cue for cue in cue_to_index.keys()]
outcomes_to_keep = [outcome for outcome in outcome_to_index.keys()]
# Train set 
filter_event_file(events_train_path,
                    filtered_events_train_path,
                    number_of_processes = num_threads,
                    keep_cues = cues_to_keep,
                    keep_outcomes = outcomes_to_keep)
# Validation set
filter_event_file(events_valid_path,
                    filtered_events_valid_path,
                    number_of_processes = num_threads,
                    keep_cues = cues_to_keep,
                    keep_outcomes = outcomes_to_keep) 

# Initialise the lists where we will store the performance scores in each epoch for the different metrics
# train
acc_hist = []
precision_hist = []
recall_hist = []
f1score_hist = []
# valid
val_acc_hist = []
val_precision_hist = []
val_recall_hist = []
val_f1score_hist = []

# Train ndl to get the weight matrix 
weights = ndl(events = filtered_events_train_path,
            alpha = params['lr'], 
            betas = (1, 1),
            method = "openmp",
            weights = weights,
            number_of_threads = num_threads,
            remove_duplicates = True,
            temporary_directory = temp_dir,
            verbose = False)

 # Predicted outcomes from the activations
y_train_pred = ev.predict_outcomes_NDL(events_path = filtered_events_train_path, 
                                    weights = weights, 
                                    chunksize = chunksize, 
                                    num_threads = num_threads)
y_valid_pred = ev.predict_outcomes_NDL(events_path = filtered_events_valid_path, 
                                    weights = weights, 
                                    chunksize = chunksize, 
                                    num_threads = num_threads)

### True outcomes 
# tain set 
events_train_df = pd.read_csv(filtered_events_train_path, header = 0, sep='\t', quotechar='"')
y_train_true = events_train_df['outcomes'].tolist()    
# validation set 
events_valid_df = pd.read_csv(filtered_events_valid_path, header = 0, sep='\t', quotechar='"')
y_valid_true = events_valid_df['outcomes'].tolist()

# Compute performance scores for the different metrics
# accuracy
acc_j = accuracy_score(y_train_true, y_train_pred)
acc_hist.append(acc_j)
val_acc_j = accuracy_score(y_valid_true, y_valid_pred)
val_acc_hist.append(val_acc_j)

# precision
precision_j = precision_score(y_train_true, y_train_pred, average = 'micro')
val_precision_j = precision_score(y_valid_true, y_valid_pred, average = 'micro')

# recall
recall_j = recall_score(y_train_true, y_train_pred, average = 'micro')
val_recall_j = recall_score(y_valid_true, y_valid_pred, average = 'micro')

# F1-score
f1score_j = f1_score(y_train_true, y_train_pred, average = 'micro')
val_f1score_j = f1_score(y_valid_true, y_valid_pred, average = 'micro')

