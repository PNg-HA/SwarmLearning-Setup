import tensorflow as tf
import logging
import numpy as np
import random as rn
import os
from util_functions import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from util_functions import *
from swarmlearning.tf import SwarmCallback
# Seed Random Numbers
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
batchSize = 128
default_max_epochs = 5
default_min_peers = 2
num_unique_classes = 19

dataDir = os.getenv('DATA_DIR', '/platform/data')
data_path = os.path.join(dataDir, '10t-10n-DOS2019-dataset-train.hdf5')
# test_path = os.path.join(dataDir, 'test.csv')
val_path = os.path.join(dataDir, '10t-10n-DOS2019-dataset-val.hdf5')
print ("data path: ", data_path)
print ("model: DNN")
max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
os.makedirs(scratchDir, exist_ok=True)
model_name = 'Lucid'

def Conv2DModel(input_shape, kernel_col, kernels=64, kernel_rows=3, learning_rate=0.001, regularization=None, dropout=None):
    # K.clear_session()

    model = Sequential()
    regularizer = regularization

    model.add(Conv2D(kernels, (kernel_rows, kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularizer, name='conv0'))
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Activation('relu'))

    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', name='fc1'))

    print(model.summary())
    compileModel(model, learning_rate)
    return model

def compileModel(model, lr):
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# subfolders = glob.glob(train_path + "/*/")
# if len(subfolders) == 0:  # only one folder case
#     subfolders = [train_path + "/"]
# else:
#     subfolders = sorted(subfolders)

# for full_path in subfolders:
#     full_path = full_path.replace("//", "/")
#     folder = full_path.split("/")[-2]
X_train, Y_train = load_dataset(data_path)
X_val, Y_val = load_dataset(val_path)

X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

# train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
# filename = train_file.split('/')[-1].strip()
# time_window = int(filename.split('-')[0].strip().replace('t', ''))
# max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
# dataset_name = filename.split('-')[2].strip()

# model_name = dataset_name + "-LUCID"
model = Conv2DModel(input_shape=X_train.shape[1:], kernel_col=X_train.shape[2])

# Create Swarm callback
swarmCallback = SwarmCallback(syncFrequency=100,
                                minPeers=min_peers,
                                mergeMethod="mean",
                                useAdaptiveSync=False,
                                adsValData=(X_val, Y_val),
                                adsValBatchSize=128)
swarmCallback.logger.setLevel(logging.DEBUG)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
# best_model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
# mc = ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, Y_train, epochs=default_max_epochs, batch_size=batchSize, validation_data=(X_val, Y_val), callbacks=[swarmCallback])

# best_model = model
# best_model.save(best_model_filename + '.h5')

swarmCallback.logger.info('Saving the final Swarm model ...')
model_path = os.path.join(scratchDir, model_name)
model.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

Y_pred_val = (model.predict(X_val) > 0.5)
Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
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

print("Model training completed.")
# print("Best model path: ", best_model_filename)
print("F1 Score of the best model on the validation set: ", f1_score_val)