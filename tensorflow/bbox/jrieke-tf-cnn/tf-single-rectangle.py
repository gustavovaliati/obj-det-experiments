from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse

from jrieke_tf_dataset import JriekeBboxDataset
from metrics import Metrics

tf.logging.set_verbosity(tf.logging.INFO)

def conv_net_02(x, n_outputs, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        # cnn from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

        '''
        Removed the max_pooling2d layeres.
        mean_IOU increased from 0.7805 to 0.8731. But got almost 10 times slower to train.
        '''
        x = tf.reshape(x, shape=[-1, 8, 8, 1])
        conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(conv1, 64, 2, activation=tf.nn.relu)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_outputs)
    return out

def conv_net_01(x, n_outputs, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        # cnn from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

        x = tf.reshape(x, shape=[-1, 8, 8, 1])
        conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 2, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_outputs)
    return out

def cnn_model_fn(features, labels, mode):

    logits_train = conv_net_01(x=features, n_outputs=4, dropout=0.25, reuse=False, is_training=True)
    logits_test = conv_net_01(x=features, n_outputs=4, dropout=0.25, reuse=True, is_training=False)

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
    train_data, train_bboxes, test_data, test_bboxes = dataset.generate()

    # Show the shapes
    print(
        'Shapes: ',
        'train_data', train_data.shape,
        'train_bboxes', train_bboxes.shape,
        'test_data', test_data.shape,
        'test_bboxes', test_bboxes.shape)


    # Show a random sample from the dataset.
    # dataset.show_generated()

    # Show random samples from the returned sets.

    # rand_index = np.random.randint(0, len(train_data)-1)
    # dataset.plot_rectangle(
    #     train_data[rand_index],
    #     dataset.convertDefaultAnnotToCoord(train_bboxes[rand_index])
    # )
    #
    # rand_index = np.random.randint(0, len(test_data)-1)
    # dataset.plot_rectangle(
    #     test_data[rand_index],
    #     dataset.convertDefaultAnnotToCoord(test_bboxes[rand_index])
    # )

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

    if not CHECKPOINT_PATH:
        print("Training...")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_bboxes,
            batch_size=100,
            num_epochs=1,
            shuffle=True)

        estimator.train(
            input_fn=train_input_fn,
            steps=None,
            # hooks=[logging_hook]
            )

    # Evaluate

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_data,
        y=test_bboxes,
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
    # pred_bboxes = [ [r] for r in predict_results]
    '''
    Transforms the output from single coordinates (only 1 obj for each image)
        [
            [1,1,1,1], # coord for image number 1
            [2,2,2,2]  # coord for image number 2
        ]
    into an array that supports multiple coordinates for each image
        [
            [ [1,1,1,1] , [2,2,2,2] ], # image number 1 (e.g., two coord)
            [ [3,3,3,3] , [4,4,4,4] ]  # image number 2 (e.g., two coord)
        ]

    '''
    pred_bboxes = np.array(list(predict_results))
    print('pred_bboxes original shape',pred_bboxes.shape)
    pred_bboxes = pred_bboxes.reshape(-1,1,4) # number of images, number of bboxes per image, number of coords
    print('pred_bboxes new shape',pred_bboxes.shape)
    print('pred_bboxes output sample', pred_bboxes[0])

    # dataset.show_predicted(pred_bboxes)

    '''
    Keras:
     - 6s - loss: 0.0248 - val_loss: 6.0943e-04
    '''

    summed_IOU = 0.
    for pred_bbox, test_bbox in zip(pred_bboxes, test_bboxes):
        # print(pred_bbox[0], test_bbox)
        summed_IOU += dataset.IOU(pred_bbox[0], test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    print('mean_IOU:',mean_IOU)


    summed_IOU = 0.
    for pred_bbox, test_bbox in zip(pred_bboxes, test_bboxes):
        # print(pred_bbox[0], test_bbox)
        summed_IOU += dataset.bbox_iou(pred_bbox[0], test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    print('mean_IOU (bbox_iou):',mean_IOU)

    pred_labels = np.full((pred_bboxes.shape[0],1),0)
    pred_scores = np.full((pred_bboxes.shape[0],1),0)
    gt_bboxes = test_bboxes.reshape(-1,1,4)
    gt_labels = pred_labels
    # print(pred_bboxes.shape,pred_labels.shape,pred_scores.shape,gt_bboxes.shape,gt_labels.shape)
    data = (pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels)
    metrics = Metrics(data)
    metrics.calc()



if __name__ == "__main__":
    tf.app.run()
