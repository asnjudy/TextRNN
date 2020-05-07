import re
import numpy as np
import os


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(pos_file, neg_file):
    ## 加载数据

    positive_examples = list(open(pos_file, 'r', encoding='utf8').readlines())
    negative_examples = list(open(neg_file, 'r', encoding='utf8').readlines())

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # generate labels
    positive_labels = [[1, 0] for _ in positive_examples]
    negative_labels = [[0, 1] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return x_text, y


def train_test_split(x, y, test_percent=0.1):
    # 打乱顺序
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    test_num_samples = int(test_percent * len(y))
    x_train, x_test = x_shuffled[:-test_num_samples], x_shuffled[-test_num_samples:]
    y_train, y_test = y_shuffled[:-test_num_samples], y_shuffled[-test_num_samples:]

    print('train/test split: {}, {}'.format(len(y_train), len(y_test)))
    return (x_train, y_train), (x_test, y_test)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)

    num_batches_of_one_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # print("*** epoch {} ***".format(epoch))
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        # 取数据
        for batch_id in range(num_batches_of_one_epoch):
            start_index = batch_id * batch_size
            end_index = (batch_id + 1) * batch_size
            end_index = min(end_index, data_size)
            yield shuffle_data[start_index:end_index]


def text_to_word_ids(x_text, vocab_path, sequence_length):
    import tensorflow as tf
    from tensorflow.contrib import learn

    if os.path.exists(vocab_path):
        tf.logging.info(">>>> restore the vocabulary from: {}".format(vocab_path))
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_word_ids = np.array(list(vocab_processor.transform(x_text)))
    else:
        vocab_processor = learn.preprocessing.VocabularyProcessor(sequence_length)
        # 把"由单词组成的文本"转换为 由单词索引构成的列表
        x_word_ids = np.array(list(vocab_processor.fit_transform(x_text)))
        tf.logging.info(">>>> write the vocabulary to: {}".format(vocab_path))
        vocab_processor.save(vocab_path)
    return x_word_ids, vocab_processor.vocabulary_


#############################
## TEST
def test_validate_x_text_and_word_ids(x_text, x, vocabulary_):
    """ 对照一下，转换前的由单词组成的文本 与 转换后的由单词索引构成的列表 是否能够相关转换
    """
    for i, word_idxs in enumerate(x[:3]):
        print(word_idxs)
        print(' '.join([vocabulary_.reverse(word_id) for word_id in word_idxs]))
        print(x_text[i])


def test_print_(vocabulary_):
    # 打印词表
    # vocab_processor.vocabulary_
    print({vocabulary_.reverse(word_id): word_id for word_id in range(len(vocabulary_))})


def test_array_index():
    arr1 = np.arange(10)
    print(arr1)

    # 从前面取 test
    print('test:', arr1[:2])
    print('train:', arr1[2:])
    # 先取 train，剩余的做 test
    print('test:', arr1[-2:])
    print('train:', arr1[:-2])
    """ outputs:
    [0 1 2 3 4 5 6 7 8 9]
    test: [0 1]
    train: [2 3 4 5 6 7 8 9]
    test: [8 9]
    train: [0 1 2 3 4 5 6 7]
    """


def test_batch_iter():
    list_a = np.arange(22)
    print(list_a)

    for batch in batch_iter(list_a, batch_size=5, num_epochs=2, shuffle=False):
        print('batch:', batch)
    """ 输出：
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
    batch: [0 1 2 3 4]
    batch: [5 6 7 8 9]
    batch: [10 11 12 13 14]
    batch: [15 16 17 18 19]
    batch: [20 21]
    batch: [0 1 2 3 4]
    batch: [5 6 7 8 9]
    batch: [10 11 12 13 14]
    batch: [15 16 17 18 19]
    batch: [20 21]
    """


def test_batch_iter2():
    x = np.arange(22)
    y = x * 10
    print(x)
    print(y)

    data = list(zip(x, y))
    print(data)

    for i, batch in enumerate(batch_iter(data=data, batch_size=5, num_epochs=1, shuffle=False)):
        print('batch %d:' % i, batch)
    """输出：
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
    [  0  10  20  30  40  50  60  70  80  90 100 110 120 130 140 150 160 170 180 190 200 210]
    [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80), (9, 90), (10, 100), (11, 110), (12, 120), (13, 130), (14, 140), (15, 150), (16, 160), (17, 170), (18, 180), (19, 190), (20, 200), (21, 210)]
    batch 0: [[ 0  0]
    [ 1 10]
    [ 2 20]
    [ 3 30]
    [ 4 40]]
    batch 1: [[ 5 50]
    [ 6 60]
    [ 7 70]
    [ 8 80]
    [ 9 90]]
    batch 2: [[ 10 100]
    [ 11 110]
    [ 12 120]
    [ 13 130]
    [ 14 140]]
    batch 3: [[ 15 150]
    [ 16 160]
    [ 17 170]
    [ 18 180]
    [ 19 190]]
    batch 4: [[ 20 200]
    [ 21 210]]
    """


def test_batch_iter3(x_test, y_test):
    test_dataset = list(zip(x_test, y_test))  # 合并到一起，一个元素代表一个样本
    test_batch_iter = batch_iter(data=test_dataset, batch_size=2, num_epochs=1, shuffle=False)
    for step, batch in enumerate(test_batch_iter):
        print('step', step)
        print('  x:', batch[:, 0])
        print('  y:', batch[:, 1])
        if step == 2:
            break


def test_batch_iter4(x_test, y_test):
    test_dataset = list(zip(x_test, y_test))
    test_batch_iter = batch_iter(data=test_dataset, batch_size=5, num_epochs=1, shuffle=False)
    for step, batch in enumerate(test_batch_iter):
        batch_xs, batch_ys = zip(*batch)  # 为什么要这么用
        print('step', step)
        print('  x:', type(batch_xs))
        print('  y:', np.array(batch_ys))
        if step == 2:
            break
