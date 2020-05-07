class Config(object):
    filter_sizes = [3, 4, 5]
    num_filters = 64
    batch_size = 128
    num_epochs = 10
    eval_per_steps = 30
    save_checkpoints_steps = 100  # 生成检查点文件的步数频率
    keep_checkpoints_max = 5  # 保留检查点文件的个数
    out_dir = None

    sequence_length = 50  # 所有文档统一到同样的长度
    num_classes = 2  # 二分类
    vocab_size = 10000  # 词表大小
    embedding_dim = 128  # 词向量维度

    checkpoint_dir = None
    train_summary_dir = None
    test_summary_dir = None

    do_train = False
    do_eval = False


def load_config_from_cmd():
    import time
    import os
    import tensorflow as tf

    flags = tf.app.flags
    """
            embedding_size = 128
            filter_sizes = [3, 4, 5]
            num_filters = 64
            batch_size = 128
            num_epochs = 10
            eval_per_steps = 30
            save_checkpoints_steps = 100
            keep_checkpoints_max = 5
            out_dir = 'run/12345'
        """
    flags.DEFINE_bool("do_train", False, "Whether to run training.")
    flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
    flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embedding (default: 128)")
    flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: 3,4,5)")
    flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
    flags.DEFINE_integer("batch_size", 128, "Batch size (default: 128)")
    flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
    flags.DEFINE_integer("eval_per_steps", 30, "Evaulate model after this many steps (default: 30)")
    flags.DEFINE_integer("save_checkpoints_steps", 100, "Save model after this many steps (default: 10)")
    flags.DEFINE_integer("keep_checkpoints_max", 5, "Max number of the checkpoints to keep (default: 5)")
    flags.DEFINE_string("out_dir", "run/12345",
                        "The directory to save running outputs, or to restore previous result")

    Config.do_train = flags.FLAGS.do_train
    Config.do_eval = flags.FLAGS.do_eval
    Config.embedding_dim = flags.FLAGS.embedding_dim
    Config.filter_sizes = list(map(int, flags.FLAGS.filter_sizes.split(',')))
    Config.num_filters = flags.FLAGS.num_filters
    Config.batch_size = flags.FLAGS.batch_size
    Config.num_epochs = flags.FLAGS.num_epochs
    Config.eval_per_steps = flags.FLAGS.eval_per_steps
    Config.save_checkpoints_steps = flags.FLAGS.save_checkpoints_steps
    Config.keep_checkpoints_max = flags.FLAGS.keep_checkpoints_max

    out_dir = flags.FLAGS.out_dir.strip()
    if out_dir == '':
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'run', timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    Config.out_dir = os.path.abspath(out_dir)

    Config.checkpoint_dir = os.path.join(Config.out_dir, 'checkpoints')
    Config.train_summary_dir = os.path.join(Config.out_dir, 'summaries', 'train')
    Config.test_summary_dir = os.path.join(Config.out_dir, 'summaries', 'test')


def test_parseArgs():
    print(Config.embedding_dim)
    load_config_from_cmd()
    print(Config.embedding_dim)
