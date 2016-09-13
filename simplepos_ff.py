from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import tensorflow as tf
import numpy as np
import csv

tf.logging.set_verbosity(tf.logging.INFO)

FEATS = ["prev2","prev1","target","post1","post2"]
LABEL = "label"
COLUMNS =  FEATS + [LABEL]


validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}


def input_fn(data_set):
  feats = {k: tf.constant(data_set[k].values) for k in COLUMNS}
  labels = {LABEL : tf.constant(data_set[LABEL].values)}
  return feats, labels


def main():
  # Load datasets
  training_set = pd.read_csv("simpleposuden_test.conll",delimiter="\t| ",names=COLUMNS, header=None,quoting=csv.QUOTE_NONE)
  test_set = pd.read_csv("simpleposuden_dev.conll",delimiter="\t| ", header=None, names=COLUMNS,quoting=csv.QUOTE_NONE)

  feature_cols = [tf.contrib.layers.sparse_column_with_hash_bucket(k,hash_bucket_size=1000) for k in FEATS]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                              hidden_units=[40, 40],
                                              n_classes=17,
                                              model_dir="/tmp/posuden",
                                              )

  with

  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      input_fn=input_fn(test_set),
      every_n_steps=50,
      metrics=validation_metrics,
      early_stopping_metric="loss",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=200)

  classifier.fit(input_fn=lambda: input_fn(training_set),
                 steps=200,monitors=[validation_monitor])


if __name__ == "__main__":
    main()