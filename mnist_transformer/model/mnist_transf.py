import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Reshape, Flatten, LayerNormalization, Permute
from tensorflow.keras.callbacks import EarlyStopping

from swarmlearning.tf import SwarmCallback

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss


batchSize = 128
default_max_epochs = 5
default_min_peers = 2

dataDir = os.getenv('DATA_DIR', '/platform/data')
data_path = os.path.join(dataDir, 'original_cardio_train.csv')
test_path = os.path.join(dataDir, 'test.csv')
val_path = os.path.join(dataDir, 'val.csv')
print ("data path: ", data_path)
print ("model: Transformer")
max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
os.makedirs(scratchDir, exist_ok=True)
model_name = 'Transf_case1_mean'

# Read the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the input to have a channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Parameters
patch_size = 7
num_patches = (28 // patch_size) ** 2
hidden_dim = 128
tokens_mlp_dim = 256
channels_mlp_dim = 128
num_blocks = 2

# Create the MLP-Mixer model
inputs = Input(shape=(28, 28, 1))
x = Reshape((num_patches, patch_size*patch_size))(inputs)

for _ in range(num_blocks):
    # Token-mixing MLP
    y = LayerNormalization()(x)
    y = Permute((2, 1))(y)
    y = Dense(tokens_mlp_dim, activation='gelu')(y)
    y = Dense(num_patches, activation='gelu')(y)
    y = Permute((2, 1))(y)
    x = x + y

    # Channel-mixing MLP
    y = LayerNormalization()(x)
    y = Dense(channels_mlp_dim, activation='gelu')(y)
    y = Dense(patch_size*patch_size, activation='gelu')(y)
    x = x + y

x = LayerNormalization()(x)
x = Flatten()(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

swarmCallback = SwarmCallback(syncFrequency=1024,
                                minPeers=min_peers,
                                mergeMethod="mean",
                                useAdaptiveSync=False,
                                adsValData=(X_test, y_test),
                                adsValBatchSize=128)
swarmCallback.logger.setLevel(logging.DEBUG)

# Train the model
model.fit(
    X_train, y_train,
    epochs = max_epochs, 
    batch_size=batchSize, 
    validation_data=(X_test,y_test), 
    callbacks=[swarmCallback]
)


swarmCallback.logger.info('Saving the final Swarm model ...')
model_path = os.path.join(scratchDir, model_name)
model.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

swarmCallback.logger.info('Starting inference on the test data ...')

# Make predictions
vit_y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Classification report
print("\nClassification Report for ViT:")
print(classification_report(y_test, vit_y_pred))