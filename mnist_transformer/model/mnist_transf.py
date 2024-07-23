import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
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
model_name = 'Transf_mean'

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the Transformer model components
num_classes = 10
input_shape = (28, 28)

inputs = Input(shape=input_shape)

# Encoder part
encoder = Reshape((28, 28))(inputs)
encoder = MultiHeadAttention(num_heads=2, key_dim=2)(encoder, encoder)
encoder = LayerNormalization()(encoder)
encoder = Flatten()(encoder)
encoder = Dropout(0.3)(encoder)

# Decoder part
decoder = Reshape((28, 28))(encoder)
decoder = MultiHeadAttention(num_heads=2, key_dim=2)(decoder, decoder)
decoder = LayerNormalization()(decoder)
decoder = Flatten()(decoder)
decoder = Dropout(0.3)(decoder)
decoder = Dense(64, activation='relu')(decoder)
decoder = Dropout(0.3)(decoder)
decoder = Dense(num_classes, activation='softmax')(decoder)

# Build the model
model = Model(inputs=inputs, outputs=decoder)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.003), loss='categorical_crossentropy', metrics=['accuracy'])


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
    epochs = 20, 
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
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)