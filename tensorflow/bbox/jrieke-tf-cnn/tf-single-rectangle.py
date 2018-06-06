from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse

from jrieke_tf_dataset import JriekeBboxDataset

tf.logging.set_verbosity(tf.logging.INFO)

def conv_net(x, n_outputs, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 8, 8, 1])
        # input ?x8x8 | kernel 5
        conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
        # input ?x5x5x32 | stride 2 kernel 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # input ?x | kernel 2
        conv2 = tf.layers.conv2d(conv1, 64, 2, activation=tf.nn.relu)
        # input ?x | stride 2 kernel 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_outputs)
    return out

def cnn_model_fn(features, labels, mode):

    logits_train = conv_net(x=features, n_outputs=4, dropout=0.25, reuse=False, is_training=True)
    logits_test = conv_net(x=features, n_outputs=4, dropout=0.25, reuse=True, is_training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=logits_test)

    # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #             logits=logits_train, labels=tf.cast(labels, dtype=tf.int32) ))
    loss_op = tf.losses.mean_squared_error(labels=labels, predictions=logits_train)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=logits_test)

    estimator_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=logits_test,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy':acc_op}
    )

    return estimator_specs


def main(unused_argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint",
                    required = False,
                    default = False,
                    help = "Load tensorflow checkpoint.")
    args = ap.parse_args()

    CHECKPOINT_PATH = None
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint

    print("TensorFlow version: {}".format(tf.__version__))

    dataset = JriekeBboxDataset()
    train_data, train_labels, eval_data, eval_labels = dataset.generate()

    # Show the shapes
    print(
        'Shapes: ',
        'train_data', train_data.shape,
        'train_labels', train_labels.shape,
        'eval_data', eval_data.shape,
        'eval_labels', eval_labels.shape)


    # Show a random sample from the dataset.
    # dataset.show_generated()

    # Show random samples from the returned sets.

    # rand_index = np.random.randint(0, len(train_data)-1)
    # dataset.plot_rectangle(
    #     train_data[rand_index],
    #     dataset.convertDefaultAnnotToCoord(train_labels[rand_index])
    # )
    #
    # rand_index = np.random.randint(0, len(eval_data)-1)
    # dataset.plot_rectangle(
    #     eval_data[rand_index],
    #     dataset.convertDefaultAnnotToCoord(eval_labels[rand_index])
    # )

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

    if not CHECKPOINT_PATH:
        print("Training...")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_labels,
            batch_size=100,
            num_epochs=20,
            shuffle=True)

        estimator.train(
            input_fn=train_input_fn,
            steps=None,
            # hooks=[logging_hook]
            )

    # Evaluate

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        y=eval_labels,
        # num_epochs=10,
        shuffle=False)

    eval_results = estimator.evaluate(
        input_fn=eval_input_fn,
        checkpoint_path=CHECKPOINT_PATH
        )
    print(eval_results)

    # Predict

    predict_results = estimator.predict(
        input_fn=eval_input_fn,
        # checkpoint_path=CHECKPOINT_PATH #this is suspect
        )

    # for r in predict_results:
        # print(r)

    pred_bboxes = [ [r] for r in predict_results]
    print('# of predicted images',len(pred_bboxes))
    dataset.show_predicted(pred_bboxes)

    '''
    Keras:
     - 6s - loss: 0.0248 - val_loss: 6.0943e-04
    '''

    summed_IOU = 0.
    for pred_bbox, test_bbox in zip(pred_bboxes, eval_labels):
        summed_IOU += dataset.IOU(pred_bbox[0], test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    print('mean_IOU:',mean_IOU)



if __name__ == "__main__":
    tf.app.run()
