# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script for training GCN models with multi-GPU support."""

import time
import models
import numpy as np
import partition_utils
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()

print(tf.__version__)
print(tf.test.is_built_with_cuda())  # Check if TensorFlow is built with CUDA
print(tf.config.list_physical_devices('GPU'))  # Check if GPUs are visible

import utils
from tensorflow.python.client import device_lib

import time

tf.logging.set_verbosity(tf.logging.INFO)
# Set random seed
seed = 1
np.random.seed(seed)

# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('save_name', './mymodel.ckpt', 'Path for saving model')
flags.DEFINE_string('dataset', 'ppi', 'Dataset string.')
flags.DEFINE_string('data_prefix', 'data/', 'Datapath prefix.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('num_clusters', 50, 'Number of clusters.')
flags.DEFINE_integer('bsize', 1, 'Number of clusters for each batch.')
flags.DEFINE_integer('num_clusters_val', 5,
                     'Number of clusters for validation.')
flags.DEFINE_integer('num_clusters_test', 1, 'Number of clusters for test.')
flags.DEFINE_integer('num_layers', 5, 'Number of GCN layers.')
flags.DEFINE_float(
    'diag_lambda', 1,
    'A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement'
)
flags.DEFINE_bool('multilabel', True, 'Multilabel or multiclass.')
flags.DEFINE_bool('layernorm', True, 'Whether to use layer normalization.')
flags.DEFINE_bool(
    'precalc', False,
    'Whether to pre-calculate the first layer (AX preprocessing).')
flags.DEFINE_bool('validation', True,
                  'Print validation accuracy after each epoch.')


def load_data(data_prefix, dataset_str, precalc):
  """Return the required data formats for GCN models."""
  if dataset_str == "ogbpr" or dataset_str == "ogbpa":
    print(f"dataset: {dataset_str}")
    (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
    train_data, val_data,
    test_data) = utils.load_t10n_data(data_prefix, dataset_str)
  else:
    (num_data, train_adj, full_adj, feats, train_feats, test_feats, labels,
    train_data, val_data,
    test_data) = utils.load_graphsage_data(data_prefix, dataset_str)

  visible_data = train_data

  y_train = np.zeros(labels.shape)
  y_val = np.zeros(labels.shape)
  y_test = np.zeros(labels.shape)
  y_train[train_data, :] = labels[train_data, :]
  y_val[val_data, :] = labels[val_data, :]
  y_test[test_data, :] = labels[test_data, :]

  train_mask = utils.sample_mask(train_data, labels.shape[0])
  val_mask = utils.sample_mask(val_data, labels.shape[0])
  test_mask = utils.sample_mask(test_data, labels.shape[0])

  if precalc:
    train_feats = train_adj.dot(feats)
    train_feats = np.hstack((train_feats, feats))
    test_feats = full_adj.dot(feats)
    test_feats = np.hstack((test_feats, feats))

  return (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
          train_mask, val_mask, test_mask, train_data, val_data, test_data,
          num_data, visible_data)


# Define model evaluation function
def evaluate(sess, model, val_features_batches, val_support_batches,
             y_val_batches, val_mask_batches, val_data, placeholders):
  """Evaluate GCN model."""
  total_pred = []
  total_lab = []
  total_loss = 0
  total_acc = 0

  num_batches = len(val_features_batches)
  for i in range(num_batches):
    features_b = val_features_batches[i]
    support_b = val_support_batches[i]
    y_val_b = y_val_batches[i]
    val_mask_b = val_mask_batches[i]
    num_data_b = np.sum(val_mask_b)
    if num_data_b == 0:
      continue
    else:
      feed_dict = utils.construct_feed_dict(features_b, support_b, y_val_b,
                                            val_mask_b, placeholders)
      outs = sess.run([model.loss, model.accuracy, model.outputs],
                      feed_dict=feed_dict)

    total_pred.append(outs[2][val_mask_b])
    total_lab.append(y_val_b[val_mask_b])
    total_loss += outs[0] * num_data_b
    total_acc += outs[1] * num_data_b

  total_pred = np.vstack(total_pred)
  total_lab = np.vstack(total_lab)
  loss = total_loss / len(val_data)
  acc = total_acc / len(val_data)

  micro, macro = utils.calc_f1(total_pred, total_lab, FLAGS.multilabel)
  return loss, acc, micro, macro


def get_available_gpus():
  """Get a list of available GPU devices."""
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            if g is not None:  # Skip None gradients
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

        if len(grads) > 0:  # If there are valid gradients
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)

            v = grad_and_vars[0][1]  # The variable associated with these gradients
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        else:
            # If all gradients for this variable are None, append (None, variable)
            v = grad_and_vars[0][1]
            average_grads.append((None, v))

    return average_grads

def main(unused_argv):

  """Main function for running experiments with multi-GPU support."""
  # Load data
  print("main started")
  (train_adj, full_adj, train_feats, test_feats, y_train, y_val, y_test,
   train_mask, val_mask, test_mask, _, val_data, test_data, num_data,
   visible_data) = load_data(FLAGS.data_prefix, FLAGS.dataset, FLAGS.precalc)

  # Partition graph and do preprocessing
  if FLAGS.bsize > 1:
    start_time = time.time()
    _, parts = partition_utils.partition_graph(train_adj, visible_data,
                                               FLAGS.num_clusters)
    print(f"[Time] partition_time: {time.time() - start_time: .3f} seconds")
    parts = [np.array(pt) for pt in parts]
  else:
    (parts, features_batches, support_batches, y_train_batches,
     train_mask_batches) = utils.preprocess(train_adj, train_feats, y_train,
                                            train_mask, visible_data,
                                            FLAGS.num_clusters,
                                            FLAGS.diag_lambda)
  start_time = time.time()
  (_, val_features_batches, val_support_batches, y_val_batches,
   val_mask_batches) = utils.preprocess(full_adj, test_feats, y_val, val_mask,
                                        np.arange(num_data),
                                        FLAGS.num_clusters_val,
                                        FLAGS.diag_lambda)

  (_, test_features_batches, test_support_batches, y_test_batches,
   test_mask_batches) = utils.preprocess(full_adj, test_feats, y_test,
                                         test_mask, np.arange(num_data),
                                         FLAGS.num_clusters_test,
                                         FLAGS.diag_lambda)
  idx_parts = list(range(len(parts)))

  print(f"[Time] preprocess: {time.time() - start_time: .3f} seconds")

  # Some preprocessing
  model_func = models.GCN

  gpu_devices = get_available_gpus()
  num_gpus = len(gpu_devices)
  tf.logging.info('Number of GPUs available: {}'.format(num_gpus))


  # Define per-GPU placeholders
  placeholders_list = []
  models_list = []

  for i in range(num_gpus):
    with tf.device('/gpu:%d' % i):
      with tf.name_scope('tower_%d' % i):
        placeholders = {
            'support':
                tf.sparse_placeholder(tf.float32, name='support_%d' % i),
            'features':
                tf.placeholder(tf.float32, name='features_%d' % i),
            'labels':
                tf.placeholder(tf.float32, shape=(None, y_train.shape[1]), name='labels_%d' % i),
            'labels_mask':
                tf.placeholder(tf.int32, name='labels_mask_%d' % i),
            'dropout':
                tf.placeholder_with_default(0., shape=(), name='dropout_%d' % i),
            'num_features_nonzero':
                tf.placeholder(tf.int32, name='num_features_nonzero_%d' % i)  # helper variable for sparse dropout
        }
        #for k, v in placeholders.items():
        #    print(k)
        #    print('-'*10)
        #    print(v)
        placeholders_list.append(placeholders)

  # Build model towers
  tower_grads = []
  tower_losses = []
  tower_accuracies = []

  shared_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    for i in range(num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('tower_%d' % i):
          # Reuse variables after the first tower

          #if i > 0:
          #  tf.get_variable_scope().reuse_variables()

          # Create model
          model = model_func(
              placeholders_list[i],
              input_dim=test_feats.shape[1],
              logging=True,
              multilabel=FLAGS.multilabel,
              norm=FLAGS.layernorm,
              precalc=FLAGS.precalc,
              num_layers=FLAGS.num_layers)

          models_list.append(model)
          # Compute loss and accuracy
          loss = model.loss
          accuracy = model.accuracy

          # Compute gradients
          grads = shared_optimizer.compute_gradients(loss)

          # Add to lists
          tower_grads.append(grads)
          tower_losses.append(loss)
          tower_accuracies.append(accuracy)

  # Average gradients
  grads = average_gradients(tower_grads)

  # Apply gradients
  train_op = shared_optimizer.apply_gradients(grads)

  # Average loss and accuracy
  loss_op = tf.reduce_mean(tower_losses)
  accuracy_op = tf.reduce_mean(tower_accuracies)

  # Initialize session
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False,
                                          log_device_placement=False))
  tf.set_random_seed(seed)

  # Init variables
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  cost_val = []
  total_training_time = 0.0
  # Train model
  epoch_times = []
  for epoch in range(FLAGS.epochs):
    t = time.time()
    np.random.shuffle(idx_parts)
    if FLAGS.bsize > 1:
      (features_batches, support_batches, y_train_batches,
       train_mask_batches) = utils.preprocess_multicluster(
           train_adj, parts, train_feats, y_train, train_mask,
           FLAGS.num_clusters, FLAGS.bsize, FLAGS.diag_lambda)
      num_batches = len(features_batches)
      num_batches = num_batches - (num_batches % num_gpus)
      for batch_start in range(0, num_batches, num_gpus):
        feed_dict = {}
        for i in range(num_gpus):
          batch_idx = batch_start + i
          if batch_idx >= num_batches:
            break
          features_b = features_batches[batch_idx]

          support_b = support_batches[batch_idx]
          #coords, values, shape = support_b
          #support_b = tf.SparseTensorValue(coords.astype(np.int64), values.astype(float), shape)
          support_b = tf.SparseTensorValue(*support_b)

          y_train_b = y_train_batches[batch_idx]
          y_train_b = y_train_b.astype(np.float32)

          train_mask_b = train_mask_batches[batch_idx]
          train_mask_b = train_mask_b.astype(int)

          tower_placeholders = placeholders_list[i]

          fd = utils.construct_feed_dict(features_b, support_b, y_train_b,
                                         train_mask_b, tower_placeholders)
          fd[tower_placeholders['dropout']] = FLAGS.dropout
          feed_dict.update(fd)
          #print(type(train_mask_b))
          #print(features_b.shape)
          # Training step

        sess.run([loss_op, train_op], feed_dict=feed_dict)

        epoch_end = time.time()
    else:
      np.random.shuffle(idx_parts)
      num_batches = len(features_batches)
      for batch_start in range(0, num_batches, num_gpus):
        feed_dict = {}
        for i in range(num_gpus):
          if batch_start + i >= len(idx_parts):
            break
          pid = idx_parts[batch_start + i]
          # Use preprocessed batch data
          features_b = features_batches[pid]
          support_b = support_batches[pid]
          y_train_b = y_train_batches[pid]
          train_mask_b = train_mask_batches[pid]
          # Get the placeholders for this tower
          tower_placeholders = placeholders_list[i]
          # Construct feed dictionary for this tower
          fd = utils.construct_feed_dict(features_b, support_b, y_train_b,
                                         train_mask_b, tower_placeholders)
          fd[tower_placeholders['dropout']] = FLAGS.dropout
          # Add to the main feed_dict
          feed_dict.update(fd)
        # Training step
        outs = sess.run([train_op, loss_op, accuracy_op], feed_dict=feed_dict)

    epoch_t = time.time() - t
    total_training_time += epoch_t
    if epoch > 2:
      epoch_times.append(epoch_t)
    print_str = 'Epoch: %04d ' % (epoch + 1) + 'training time: {:.5f} '.format(
        total_training_time)

    # Validation
    if FLAGS.validation:
      cost, acc, micro, macro = evaluate(sess, models_list[0], val_features_batches,
                                         val_support_batches, y_val_batches,
                                         val_mask_batches, val_data,
                                         placeholders_list[0])
      cost_val.append(cost)
      print_str += 'val_acc= {:.5f} '.format(
          acc) + 'mi F1= {:.5f} ma F1= {:.5f} '.format(micro, macro)

    tf.logging.info(print_str)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
        cost_val[-(FLAGS.early_stopping + 1):-1]):
      tf.logging.info('Early stopping...')
      break

  tf.logging.info('Optimization Finished!')
  print(f"[Time] epoch_t {np.mean(epoch_times):.3f}({np.std(epoch_times):.3f})")
  # Save model
  saver.save(sess, FLAGS.save_name)

  """
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess, FLAGS.save_name)

  # Testing
  test_cost, test_acc, micro, macro = evaluate(
      sess, models_list[0], test_features_batches, test_support_batches,
      y_test_batches, test_mask_batches, test_data, placeholders_list[0])
  print_str = 'Test set results: ' + 'cost= {:.5f} '.format(
      test_cost) + 'accuracy= {:.5f} '.format(
          test_acc) + 'mi F1= {:.5f} ma F1= {:.5f}'.format(micro, macro)
  tf.logging.info(print_str)
  """


if __name__ == '__main__':
  print("__main__")
  tf.app.run(main)
