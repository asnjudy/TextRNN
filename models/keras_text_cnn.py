import tensorflow as tf
import numpy as np

vocabulary_size = 16700
num_samples = 200
max_document_length = 50

x_word_ids = np.random.randint(0, 16782, size=(num_samples, max_document_length))
y = np.random.randint(2, size=(num_samples, ))
y_onehot = tf.keras.utils.to_categorical(y)

model = tf.keras.Sequential()
model.add()