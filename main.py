import os
from config import Config, load_config_from_cmd
from data_process import load_data_and_labels, text_to_word_ids, train_test_split
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# 加载命令行中指定的参数
load_config_from_cmd()

neg_file = './data/rt-polarity.neg'
pos_file = './data/rt-polarity.pos'

x_text, y = load_data_and_labels(pos_file, neg_file)
Config.num_classes = y.shape[1]
# 获取文本的最大长度
max_document_length = max(len(x.split(' ')) for x in x_text)
Config.sequence_length = max_document_length

vocab_path = os.path.join(Config.out_dir, "vocab")
x_word_ids, vocabulary_ = text_to_word_ids(x_text, vocab_path, Config.sequence_length)
Config.vocab_size = len(vocabulary_)

# 划分训练集和测试集
(x_train, y_train), (x_test, y_test) = train_test_split(x_word_ids, y)

# Config.do_train = True
Config.do_eval = True


def impl1():
    from impl1.func_train_and_eval import train_and_evaluate, evaluate

    if Config.do_train:
        train_and_evaluate(x_train, y_train, x_test, y_test)
    if Config.do_eval:
        evaluate(x_text, y, x_word_ids)


def impl2():
    from impl2.text_cnn_helper import TextCNNHelper

    helper = TextCNNHelper()
    if Config.do_train:
        helper.train_and_evaluate(x_train, y_train, x_test, y_test)

    if Config.do_eval:
        helper.evaluate(x_word_ids[:100], y[:100])


def main(argv=None):
    impl2()


if __name__ == '__main__':
    tf.app.run()
