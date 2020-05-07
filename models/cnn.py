import tensorflow as tf


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.weights = []  # trainable_variables

    def __call__(self, inputs, keep_drop=1.0):
        """ inputs 输入数据的形状为:
        tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        [
            [词索引, 词索引, ....],  # 代表一个句子
            [词索引, 词索引, ....],  # 代表一个句子
            [],
            ..
        ]
        """
        with tf.name_scope('embedding'):
            embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],
                                                      minval=-1.0,
                                                      maxval=1.0),
                                    name='embedding_matrix')
            # 词索引 -> 词向量, 假设 sequence_length=100, embedding_size=256
            # (None, 100) -> (None, 100, 256)
            inputs_embedded = tf.nn.embedding_lookup(embedding, inputs)
            # 转化为当通道图像的二维卷积
            # (None, 100, 256) -> (None, 100, 256, 1)
            inputs_embedded_expanded = tf.expand_dims(inputs_embedded, -1)
            print('##', inputs)
            print('##', inputs_embedded)
            print('##', inputs_embedded_expanded)
            # 记录词向量矩阵
            self.weights.append(embedding)

        flattened_pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv_op_name = 'conv_%s' % (i + 1)
            with tf.name_scope(conv_op_name):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='kernel')
                b = tf.Variable(tf.constant(0.1, shape=(self.num_filters,)), name='bias')
                # 卷积操作，假设 filter_size=3, num_filters=64
                # (None, 100, 256, 1) -> (None, 98, 1, 64)
                conv_out = tf.nn.conv2d(inputs_embedded_expanded,
                                        W,
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name=conv_op_name)
                print('##', conv_out)
                # 加偏置并激活
                conv_out = tf.nn.relu(tf.nn.bias_add(conv_out, b))
                # 记录卷积核及偏置
                self.weights.append(W)
                self.weights.append(b)

            maxpool_op_name = 'maxpool_%s' % (i + 1)
            with tf.name_scope(maxpool_op_name):
                pool_height_size = self.sequence_length - filter_size + 1
                pool_width = 1
                # 池化操作
                # (None, 98, 1, 64) -> (None, 1, 1, 64)
                pooled_out = tf.nn.max_pool(conv_out,
                                            ksize=[1, pool_height_size, pool_width, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name=maxpool_op_name)
                print('##', pooled_out)
                # (None, 1, 1, 64) -> (None, 64)
                flattened_pooled_out = tf.reshape(pooled_out, shape=(-1, self.num_filters))
                print("** Flaten:", flattened_pooled_out)
                flattened_pooled_outputs.append(flattened_pooled_out)

        # 把所有卷积池化后的结构拼接为一个长的向量作为全连接层的输入
        fc1_in = tf.concat(flattened_pooled_outputs, axis=1)
        print("## fc1 inputs:", fc1_in)

        with tf.name_scope('fc1'):
            input_units = fc1_in.shape[1].value
            print("fc1 input units:", input_units)
            W = tf.Variable(tf.truncated_normal((input_units, self.num_classes), stddev=0.1), name='kernel')
            b = tf.Variable(tf.constant(0.1, shape=(self.num_classes,)), name='bias')
            fc1_out = tf.nn.bias_add(tf.matmul(fc1_in, W), b)
            print("## fc1 outputs:", fc1_out)
            self.weights.append(W)
            self.weights.append(b)
        # 记录模型的输出 Tensor, 即各类别的打分数据，在调用处，用于优化损失
        self.output = fc1_out
        return self.output


def cnn_test():
    text_cnn = TextCNN(sequence_length=100,
                       num_classes=3,
                       vocab_size=10000,
                       embedding_size=256,
                       filter_sizes=[3, 4, 5],
                       num_filters=64)

    x = tf.placeholder(tf.int32, [2, 100], name='input_x')
    output = text_cnn(x)
    print(output)

    print("trainable_variables:")
    for weight in text_cnn.weights:
        print(weight)
