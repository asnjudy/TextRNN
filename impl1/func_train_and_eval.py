import os
from data_process import load_data_and_labels, batch_iter, text_to_word_ids, train_test_split
from models.cnn import TextCNN
import tensorflow as tf
from config import Config, load_config_from_cmd


def train_and_evaluate(x_train, y_train, x_test, y_test):
    text_cnn = TextCNN(Config.sequence_length, Config.num_classes, Config.vocab_size, Config.embedding_dim,
                       Config.filter_sizes, Config.num_filters)

    x = tf.placeholder(tf.int32, [None, Config.sequence_length], name='input_x')
    y_ = tf.placeholder(tf.float32, [None, Config.num_classes], name='input_y')

    output = text_cnn(x)

    # 交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=output))

    predictions = tf.argmax(output, axis=1, name='predictions')
    correction_predictions = tf.equal(predictions, tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correction_predictions, dtype=tf.float32))

    # 选择梯度优化算法
    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step)

    with tf.Session() as sess:
        loss_summary = tf.summary.scalar('loss', loss)
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
        summary_op = tf.summary.merge([loss_summary, accuracy_summary])

        train_summary_writer = tf.summary.FileWriter(Config.train_summary_dir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(Config.test_summary_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=Config.keep_checkpoints_max)
        sess.run(tf.global_variables_initializer())

        all_variables = tf.get_default_graph().get_collection('variables')
        print('all variables:\n', all_variables)
        trainable_variables = tf.get_default_graph().get_collection('trainable_variables')
        print('trainable variables:\n', trainable_variables)

        if not os.path.exists(Config.checkpoint_dir):
            os.makedirs(Config.checkpoint_dir)
        else:
            # 加载检查点文件
            checkpoint_file = tf.train.latest_checkpoint(Config.checkpoint_dir)
            saver.restore(sess, checkpoint_file)
            tf.logging.info("<<<< restore model from:", checkpoint_file)

        train_dataset = list(zip(x_train, y_train))
        train_batch_iter = batch_iter(train_dataset, Config.batch_size, num_epochs=100, shuffle=False)

        for batch in train_batch_iter:
            batch_xs, batch_ys = zip(*batch)
            _, _train_loss, _train_acc, _train_summary = sess.run([train_op, loss, accuracy, summary_op],
                                                                  feed_dict={
                                                                      x: batch_xs,
                                                                      y_: batch_ys
                                                                  })
            step = tf.train.global_step(sess, global_step)
            train_summary_writer.add_summary(_train_summary, step)
            # print("step {}, loss {:g}, acc {:g}".format(step, _train_loss, _train_acc))

            # 在测试集上计算模型的准确度
            if step % Config.eval_per_steps == 0:
                _test_acc, _test_loss, _test_summary = sess.run([accuracy, loss, summary_op],
                                                                feed_dict={
                                                                    x: x_test,
                                                                    y_: y_test
                                                                })

                print('step {} - loss: {:.4f}, aac: {:.4f}, test loss: {:.4f}, test acc: {:.4f}'.format(
                    step, _train_loss, _train_acc, _test_loss, _test_acc))
                test_summary_writer.add_summary(_train_summary, step)

            # 保存模型的权重
            if step % Config.save_checkpoints_steps == 0:
                checkpoint_prefix = os.path.join(Config.checkpoint_dir, 'model')
                # global_step 拼接到 checkpoint_prefix 后面
                # checkpoint_prefix-global_step.(index|meta|data-00000-of-00001)
                save_path = saver.save(sess, checkpoint_prefix, global_step=step)
                tf.logging.info(">>>> save model in file: {}".format(save_path))


def evaluate(x_text, y, x_word_ids):
    import numpy as np
    import csv

    with tf.Session() as sess:
        # Load the saved meta graph
        checkpoint_file = tf.train.latest_checkpoint(Config.checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # Restore variables
        saver.restore(sess, checkpoint_file)

        graph = tf.get_default_graph()
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        predictions = graph.get_operation_by_name("predictions").outputs[0]

        batches = batch_iter(list(x_word_ids), batch_size=128, num_epochs=1, shuffle=False)
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

    correct_predictions = sum(all_predictions == np.argmax(y, axis=1))

    print("Total number of test examples: {}".format(len(y)))
    print("Accuracy: {:g}".format(float(correct_predictions) / float(len(y))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_text), all_predictions))
    out_path = os.path.join("./prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
