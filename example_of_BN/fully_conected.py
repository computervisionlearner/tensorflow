from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import datasets_mnist
import mnist

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  phase_pl = tf.placeholder(tf.bool)
  return images_placeholder, labels_placeholder, phase_pl


def fill_feed_dict(data_set, images_pl, labels_pl, phase_pl, phase):

  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      phase_pl: phase
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            phase_pl,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder, phase_pl, False)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  
  
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return precision

def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  train ,validation,test = datasets_mnist.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder, phase_pl = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2, phase_pl)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)
    tf.summary.scalar('loss', loss)
    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()


    g_list = tf.global_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(var_list=g_list)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(train,
                                 images_placeholder,
                                 labels_placeholder, phase_pl, True)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 10 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        train_precision = do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                phase_pl,
                train)
        tmp1 = tf.Summary(value=[tf.Summary.Value(tag="precision/Training",
                                                               simple_value=train_precision)])
        summary_writer.add_summary(tmp1, step)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        val_precision = do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                phase_pl,
                validation)
        tmp2 = tf.Summary(value=[tf.Summary.Value(tag="precision/Validation",
                                                               simple_value=val_precision)])
        summary_writer.add_summary(tmp2, step)
        # Evaluate against the test set.
        print('Test Data Eval:')
        test_precision = do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                phase_pl,
                test)
        tmp3 = tf.Summary(value=[tf.Summary.Value(tag="precision/Test",
                                                               simple_value=test_precision)])
        summary_writer.add_summary(tmp3, step)
        
        summary_writer.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--max_steps',
      type=int,
      default=2010,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=500,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default=os.path.join('datasets'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('log'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()