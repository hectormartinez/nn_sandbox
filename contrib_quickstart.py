from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
POS_TRAIN = "simpleposuden_train.conll"
POST_TEST = "simpleposuden_test.conll"
POST_DEV = "simpleposuden_dev.conll"

# Load datasets.


# Specify that all features have multinomial_data

label = tf.contrib.layers.sparse_column_with_keys(
  column_name="label", keys=["NOUN", "ADJ", "VERB"])


feature_columns = [tf.contrib.layers.sparse_column_with_hash_bucket("", dimension=5)]

validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    metrics=validation_metrics,
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/posuden",
                                            config=tf.contrib.learn.RunConfig(
                                            save_checkpoints_secs=1))

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.