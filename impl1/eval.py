import os
from data_process import load_data_and_labels, batch_iter
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import csv
from config import Config

neg_file = '../data/rt-polarity.neg'
pos_file = '../data/rt-polarity.pos'

x_text, y = load_data_and_labels(pos_file, neg_file)

vocab_path = os.path.join(Config.out_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_word_ids = np.array(list(vocab_processor.transform(x_text)))

with tf.Session() as sess:
    # Load the saved meta graph
    saver = tf.train.import_meta_graph("{}.meta".format(Config.checkpoint_dir))
    # Restore variables
    saver.restore(sess, Config.checkpoint_dir)

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
