import os
import tensorflow as tf
from data_process import load_data_and_labels, text_to_word_ids, batch_iter
from config import Config, load_config_from_cmd
from models.cnn import TextCNN


class TextCNNHelper(object):
    def __init__(self):

        self.x = tf.placeholder(tf.int32, [None, Config.sequence_length], name='input_x')
        self.y_ = tf.placeholder(tf.float32, [None, Config.num_classes], name='input_y')

        self.net = TextCNN(Config.sequence_length, Config.num_classes, Config.vocab_size, Config.embedding_dim,
                           Config.filter_sizes,
                           Config.num_filters)
        # 前向计算，获得模型输出
        output = self.net(self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=output))
        self.predictions = tf.argmax(output, axis=1, name='predictions')
        correction_predictions = tf.equal(self.predictions, tf.argmax(self.y_, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correction_predictions, dtype=tf.float32))

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        global_step = tf.Variable(0, trainable=False)
        train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step)
        # summaries
        loss_summary = tf.summary.scalar('loss', self.loss)
        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        with tf.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(Config.train_summary_dir, sess.graph)
            test_summary_writer = tf.summary.FileWriter(Config.test_summary_dir, sess.graph)

            saver = tf.train.Saver(max_to_keep=Config.keep_checkpoints_max)

            # 初始化所有变量
            sess.run(tf.global_variables_initializer())

            if not os.path.exists(Config.checkpoint_dir):
                os.makedirs(Config.checkpoint_dir)
            else:
                checkpoint_file = tf.train.latest_checkpoint(Config.checkpoint_dir)
                if checkpoint_file is not None:
                    # 加载检查点文件
                    saver.restore(sess, checkpoint_file)
                    tf.logging.info("<<<< restore model from: {}".format(checkpoint_file))

            train_dataset = list(zip(x_train, y_train))
            train_batch_iter = batch_iter(train_dataset,
                                          Config.batch_size,
                                          num_epochs=Config.num_epochs,
                                          shuffle=False)
            for batch in train_batch_iter:
                batch_xs, batch_ys = zip(*batch)
                _, train_loss, train_acc, train_summary = sess.run(
                    [train_op, self.loss, self.accuracy, summary_op],
                    feed_dict={
                        self.x: batch_xs,
                        self.y_: batch_ys
                    })
                step = tf.train.global_step(sess, global_step)
                train_summary_writer.add_summary(train_summary, step)
                # print("step {}, loss {:g}, acc {:g}".format(step, _train_loss, _train_acc))
                # 保存模型检查点
                if step % Config.save_checkpoints_steps == 0:
                    checkpoint_prefix = os.path.join(Config.checkpoint_dir, 'model')
                    save_path = saver.save(sess, checkpoint_prefix, global_step=step)
                    tf.logging.info(">>>> save model in file: {}".format(save_path))

                # 在测试集上计算模型的准确度
                if step % Config.eval_per_steps == 0:
                    test_acc, test_loss, test_summary = sess.run([self.accuracy, self.loss, summary_op],
                                                                 feed_dict={
                                                                     self.x: x_test,
                                                                     self.y_: y_test
                                                                 })
                    print('step {} - loss: {:.4f}, aac: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'.format(
                        step, train_loss, train_acc, test_loss, test_acc))
                    test_summary_writer.add_summary(test_summary, step)

    def evaluate(self, x_test, y_test, out_file='./prediction.csv'):
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=Config.keep_checkpoints_max)
            checkpoint_file = tf.train.latest_checkpoint(Config.checkpoint_dir)
            saver.restore(sess, checkpoint_file)
            tf.logging.info("<<<< restore model from: {}".format(checkpoint_file))

            predictions, acc, loss, = sess.run([self.predictions, self.loss, self.accuracy],
                                               feed_dict={
                                                   self.x: x_test,
                                                   self.y_: y_test
                                               })

            print("evaluate loss {:g}, acc {:g}".format(loss, acc))

    def predict(self):
        pass
