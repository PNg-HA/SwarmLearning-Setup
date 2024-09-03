import tensorflow as tf
import logging
import numpy as np
import random as rn
import os
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.merge import concatenate

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from swarmlearning.tf import SwarmCallback
# Seed Random Numbers
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
model_name = 'multimodal_mnist'

# Load and preprocess the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the 28x28 images into vectors of size 784
x_train = x_train.reshape(-1, 784).astype('float32') / 255
x_test = x_test.reshape(-1, 784).astype('float32') / 255

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the left branch of the model
left_input = Input(shape=(784,), name='left_input')
left_branch = Dense(32, activation='relu', name='left_branch')(left_input)

# Define the right branch of the model
right_input = Input(shape=(784,), name='right_input')
right_branch = Dense(32, activation='relu', name='right_branch')(right_input)

# Merge the branches
x = concatenate([left_branch, right_branch])

# Final output layer
predictions = Dense(10, activation='softmax', name='main_output')(x)

# Define the model
model = Model(inputs=[left_input, right_input], outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Create Swarm callback
swarmCallback = SwarmCallback(syncFrequency=100,
                                minPeers=min_peers,
                                mergeMethod="mean",
                                useAdaptiveSync=False,
                                adsValData=([x_test, x_test], y_test),
                                adsValBatchSize=128)
swarmCallback.logger.setLevel(logging.DEBUG)
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
# mc = ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
model.fit([x_train, x_train], y_train, epochs=default_max_epochs, batch_size=128, validation_data=([x_test, x_test], y_test), callbacks=[swarmCallback])



swarmCallback.logger.info('Saving the final Swarm model ...')
model_path = os.path.join(scratchDir, model_name)
model.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

# Evaluate the model
loss, accuracy = model.evaluate([x_test, x_test], y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')