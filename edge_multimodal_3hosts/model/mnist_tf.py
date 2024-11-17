import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.merge import concatenate

import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalMaxPooling2D, GlobalAveragePooling1D, Flatten, Dense, Dropout, Activation, Input, concatenate
import random as rn
from tensorflow.keras.layers import Conv2D
# from sklearn.utils import shuffle
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from util_functions import *
from swarmlearning.tf import SwarmCallback
# Seed Random Numbers
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
batchSize = 1024
default_max_epochs = 10
default_min_peers = 3
num_unique_classes = 19

dataDir = os.getenv('DATA_DIR', '/platform/data')
data_path_h5 = os.path.join(dataDir, 'train.hdf5')
val_path_h5 = os.path.join(dataDir, 'val.hdf5')
test_path_h5 = os.path.join(dataDir, 'test.hdf5')
# test_path = os.path.join(dataDir, 'test.csv')
# val_path = os.path.join(dataDir, '10t-10n-DOS2019-dataset-val.hdf5')
data_path_csv = os.path.join(dataDir, 'train.csv')
val_path_csv = os.path.join(dataDir, 'val.csv')
test_path_csv = os.path.join(dataDir, 'test.csv')
print ("data path: ", data_path_h5)
print ("model: DNN")
max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
os.makedirs(scratchDir, exist_ok=True)
model_name = 'Lucid'

# subfolders = glob.glob(train_path + "/*/")
# if len(subfolders) == 0:  # only one folder case
#     subfolders = [train_path + "/"]
# else:
#     subfolders = sorted(subfolders)

# for full_path in subfolders:
#     full_path = full_path.replace("//", "/")
#     folder = full_path.split("/")[-2]

X_train_1, Y_train_1 = load_dataset(data_path_h5)
X_val_1, Y_val_1 = load_dataset(val_path_h5)
X_test_1, Y_test_1 = load_dataset(test_path_h5)

train_df = pd.read_csv(data_path_csv)
val_df = pd.read_csv(val_path_csv)
test_df = pd.read_csv(test_path_csv)

train_df = train_df.drop(train_df.columns[0], axis=1)

dummy = train_df.drop(['Attack_type', 'Attack_label'], axis=1)
X_columns = dummy.columns.tolist()
y_column = "Attack_label"

X_train_2 = train_df[X_columns]
Y_train_2 = train_df[y_column]

X_val_2 = val_df[X_columns]
Y_val_2 = val_df[y_column]

X_test_2 = test_df[X_columns]
Y_test_2 = test_df[y_column]

scaler = StandardScaler()
X_train_2[X_columns] = scaler.fit_transform(X_train_2[X_columns])
X_val_2[X_columns] = scaler.fit_transform(X_val_2[X_columns])
X_test_2[X_columns] = scaler.fit_transform(X_test_2[X_columns])

# # Reshape the dataset 1
# X_train_1 = X_1[:188135]
# X_val_1 = X_1[188135:209038]
# X_test_1 = X_1[209038:232265]

# Y_train_1 = Y_1[:188135]
# Y_val_1 = Y_1[188135:209038]
# Y_test_1 = Y_1[209038:232265]

print ("X_train_1, X_val_1, X_test_1:", X_train_1.shape, X_val_1.shape, X_test_1.shape)
# Define the first branch model
input_1 = Input(shape=X_train_1.shape[1:])
x1 = Conv2D(64, (3, X_train_1.shape[2]), strides=(1, 1), name='conv0')(input_1)
x1 = Activation('relu')(x1)
x1 = GlobalMaxPooling2D()(x1)
x1 = Flatten()(x1)
x1 = Dense(60, activation='relu')(x1)
# x1 = Dense(1, activation='sigmoid', name='fc1')(x1)


# Define the second branch model
input_2 = Input(shape=(X_train_2.shape[1], 1))
x2 = Conv1D(74, 5, activation='relu')(input_2)
x2 = Conv1D(50, 5, activation='relu')(x2)
x2 = GlobalAveragePooling1D()(x2)
x2 = Dense(60, activation='relu')(x2)
# x2 = Dropout(0.3)(x2)
# x2 = Dense(60, activation='relu')(x2)
x2 = Dropout(0.3)(x2)
x2 = Dense(60, activation='relu')(x2)



combined = concatenate([x1, x2])

# Add the final dense layer
output = Dense(1, activation='sigmoid')(combined)

# Define the complete model
model = Model(inputs=[input_1, input_2], outputs=output)

# Create Swarm callback
swarmCallback = SwarmCallback(syncFrequency=1024,
                                minPeers=min_peers,
                                mergeMethod="mean",
                                useAdaptiveSync=False,
                                adsValData=([X_val_1, X_val_2], Y_val_1),
                                adsValBatchSize=128)
swarmCallback.logger.setLevel(logging.DEBUG)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
# best_model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
# mc = ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit([X_train_1, X_train_2], Y_train_1, epochs=default_max_epochs, batch_size=batchSize, validation_data=([X_val_1, X_val_2], Y_val_1), callbacks=[swarmCallback])

# best_model = model
# best_model.save(best_model_filename + '.h5')

swarmCallback.logger.info('Saving the final Swarm model ...')
model_path = os.path.join(scratchDir, model_name)
model.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

print("Model training completed.")
# print("Best model path: ", best_model_filename)

loss, accuracy = model.evaluate([X_test_1, X_test_2], Y_test_1)
print(f'Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}')

Y_pred_val = (model.predict([X_test_1, X_test_2]) > 0.5)
Y_true_val = Y_test_1.reshape((Y_test_1.shape[0], 1))
f1_score_val = f1_score(Y_true_val, Y_pred_val)
accuracy = accuracy_score(Y_true_val, Y_pred_val)

# val_file = open(best_model_filename + '.csv', 'w', newline='')
# val_file.truncate(0)
# val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
# val_writer.writeheader()
# val_file.flush()
# row = {'Model': model_name, 'Samples': Y_pred_val.shape[0], 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1_score_val),
#        'Hyper-parameters': hyperparamters, "Validation Set": glob.glob(dataset_folder + "/*" + '-val.hdf5')[0]}
# val_writer.writerow(row)
# val_file.close()


print("F1 Score of the best model on the validation set: ", f1_score_val)

