import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Layer
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
df = pd.read_csv(data_path, sep=";")
df.head()

# Preprocess data
X = df.drop('cardio', axis=1)  # Features
y = df['cardio']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, Y_val = X_train, X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Reshape the data to add a sequence dimension
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Define the Transformer Encoder Block
class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the Vision Transformer (ViT) Model
class VisionTransformer(Model):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_layers, num_classes, rate=0.1):
        super(VisionTransformer, self).__init__()
        self.embedding = Dense(embed_dim)
        self.cls_token = self.add_weight(shape=(1, 1, embed_dim), initializer='random_normal', trainable=True)
        self.pos_embedding = self.add_weight(shape=(1, num_patches + 1, embed_dim), initializer='random_normal', trainable=True)
        self.transformer_blocks = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.mlp_head = Dense(num_classes, activation='sigmoid')

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = self.embedding(x)
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, tf.shape(x)[-1]])
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embedding
        for block in self.transformer_blocks:
            x = block(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)

# Parameters
num_patches = X_train_scaled.shape[1]
embed_dim = 64
num_heads = 4
ff_dim = 128
num_layers = 4
num_classes = 1  # Binary classification
dropout_rate = 0.1

# Instantiate and compile the model
model = VisionTransformer(num_patches, embed_dim, num_heads, ff_dim, num_layers, num_classes, dropout_rate)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])

sample_input_shape = (None, num_patches, X_train_scaled.shape[2])
model.build(sample_input_shape)

swarmCallback = SwarmCallback(syncFrequency=1024,
                                minPeers=min_peers,
                                mergeMethod="mean",
                                useAdaptiveSync=False,
                                adsValData=(X_test, y_test),
                                adsValBatchSize=128)
swarmCallback.logger.setLevel(logging.DEBUG)

# Train the model
model.fit(
    X_train_scaled, y_train,
    epochs = max_epochs, 
    batch_size=batchSize, 
    validation_data=(X_val,Y_val), 
    callbacks=[swarmCallback]
)


swarmCallback.logger.info('Saving the final Swarm model ...')
model_path = os.path.join(scratchDir, model_name)
model.save(model_path)
swarmCallback.logger.info(f'Saved the trained model - {model_path}')

swarmCallback.logger.info('Starting inference on the test data ...')

# Make predictions
vit_y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Classification report
print("\nClassification Report for ViT:")
print(classification_report(y_test, vit_y_pred))