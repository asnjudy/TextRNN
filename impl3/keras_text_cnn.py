import tensorflow as tf

layers = tf.keras.layers


class TextCNN(tf.keras.Model):
    def __init__(self, sequence_length, vocabulary_size, embedding_dim, num_classes):
        super(TextCNN, self).__init__('my_model')

        self.embedding = layers.Embedding(vocabulary_size, embedding_dim, input_length=sequence_length)
        self.conv1 = layers.Conv1D(64, 3, strides=1, padding='valid', use_bias=True, activation='relu')
        self.conv2 = layers.Conv1D(64, 4, strides=1, padding='valid', use_bias=True, activation='relu')
        self.conv3 = layers.Conv1D(64, 5, strides=1, padding='valid', use_bias=True, activation='relu')
        self.maxpool1 = layers.GlobalMaxPool1D()
        self.maxpool2 = layers.GlobalMaxPool1D()
        self.maxpool3 = layers.GlobalMaxPool1D()
        self.dense1 = layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, inputs):
        """
        (None, 52) ->  (None, 52, 128)
        """
        embedded_inputs = self.embedding(inputs)
        # 卷积并池化
        pooled_out_1 = self.maxpool1(self.conv1(embedded_inputs))
        pooled_out_2 = self.maxpool2(self.conv2(embedded_inputs))
        pooled_out_3 = self.maxpool3(self.conv3(embedded_inputs))

        # 拼接在一起
        pooled_out_all = layers.concatenate([pooled_out_1, pooled_out_2, pooled_out_3], axis=-1)
        return self.dense1(pooled_out_all)


def keras_cnn_text(sequence_length, vocabulary_size, embedding_dim, num_classes):
    text_input = tf.keras.Input(shape=(sequence_length, ), dtype='int32', name='text_input')
    embedded_text = layers.Embedding(vocabulary_size, embedding_dim)(text_input)
    conv1 = layers.Conv1D(64, 3, padding='valid', use_bias=True, activation='relu')(embedded_text)
    pooled1 = layers.GlobalMaxPool1D()(conv1)

    conv2 = layers.Conv1D(64, 4, padding='valid', use_bias=True, activation='relu')(embedded_text)
    pooled2 = layers.GlobalMaxPool1D()(conv2)

    conv3 = layers.Conv1D(64, 5, padding='valid', use_bias=True, activation='relu')(embedded_text)
    pooled3 = layers.GlobalMaxPool1D()(conv3)

    pooled_all = layers.concatenate([pooled1, pooled2, pooled3], axis=-1)

    dense1 = layers.Dense(num_classes, activation=tf.nn.softmax)(pooled_all)
    return tf.keras.Model(text_input, dense1)


def test_text_cnn():
    import numpy as np

    num_classes = 2

    embedding_dim = 128
    vocabulary_size = 16782
    num_samples = 200
    sequence_length = 52

    x_word_ids = np.random.randint(0, 16782, size=(num_samples, sequence_length))
    y = np.random.randint(2, size=(num_samples, ))
    y_one_hot = tf.keras.utils.to_categorical(y)

    # model = TextCNN(sequence_length, vocabulary_size, embedding_dim, num_classes)
    # model.build(input_shape=(sequence_length))
    model = keras_cnn_text(sequence_length, vocabulary_size, embedding_dim, num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_word_ids, y_one_hot, batch_size=100, epochs=3)
    tf.keras.utils.plot_model(model, show_shapes=True)


test_text_cnn()
