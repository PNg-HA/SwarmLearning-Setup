import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss

from swarmlearning.tf import SwarmCallback

batchSize = 128
default_max_epochs = 5
default_min_peers = 3
num_unique_classes = 19

dataDir = os.getenv('DATA_DIR', '/platform/data')
data_path = os.path.join(dataDir, 'train.csv')
test_path = os.path.join(dataDir, 'test.csv')
val_path = os.path.join(dataDir, 'val.csv')
print ("data path: ", data_path)
print ("model: CNN")
max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
os.makedirs(scratchDir, exist_ok=True)
model_name = 'CiCIoT_case1_mean'

X_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight',
]

y_column = 'label'

train_df = pd.read_csv(data_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

train_df = train_df.drop(train_df.columns[0], axis=1)
val_df = val_df.drop(val_df.columns[0], axis=1)
test_df = test_df.drop(test_df.columns[0], axis=1)

X_train = train_df[X_columns]
Y_train = train_df[y_column]

X_val = val_df[X_columns]
Y_val = val_df[y_column]

X_test = test_df[X_columns]
Y_test = test_df[y_column]

scaler = StandardScaler()
X_train[X_columns] = scaler.fit_transform(X_train[X_columns])
X_val[X_columns] = scaler.fit_transform(X_val[X_columns])
X_test[X_columns] = scaler.fit_transform(X_test[X_columns])

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

model = Sequential([
    Conv1D(74, 5, activation='relu', input_shape=(X_train.shape[1], 1)),
    Conv1D(50, 5, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(120, activation='relu'),
    Dropout(0.1),
    Dense(120, activation='relu'),
    Dropout(0.1),
    Dense(120, activation='relu'),
    Dropout(0.1),
    Dense(num_unique_classes, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create Swarm callback
swarmCallback = SwarmCallback(syncFrequency=1024,
                                minPeers=min_peers,
                                mergeMethod="mean",
                                useAdaptiveSync=False,
                                adsValData=(X_test, Y_test),
                                adsValBatchSize=128)
swarmCallback.logger.setLevel(logging.DEBUG)

model.fit(X_train, Y_train, epochs = max_epochs, batch_size=batchSize, validation_data=(X_val,Y_val), callbacks=[swarmCallback])

swarmCallback.logger.info('Saving the final Swarm model ...')
model_path = os.path.join(scratchDir, model_name)
model.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

swarmCallback.logger.info('Starting inference on the test data ...')

y_pred = model.predict(X_test)
Y_pred = tf.argmax(y_pred, axis=1)

print('Loss = ', hamming_loss(Y_test, Y_pred))
print('Accuracy = ', accuracy_score(Y_test, Y_pred))
print('Precision = ', precision_score(Y_test, Y_pred, average='micro'))
print('Recall = ', recall_score(Y_test, Y_pred, average='micro'))
print('F1 = ', f1_score(Y_test, Y_pred, average='micro'))
# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)

# Set print options
np.set_printoptions(threshold=np.inf)

print("Confusion matrix: \n", cnf_matrix)