from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse

from jrieke_tf_dataset import JriekeBboxDataset

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    print(features,labels,mode)
    flat = tf.reshape(features["x"], [-1, 8 * 8 * 1])
    dense = tf.layers.dense(inputs=flat, units=200, activation=tf.nn.relu)
    print('training mode?', mode == tf.estimator.ModeKeys.TRAIN)
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.2,
            training=mode == tf.estimator.ModeKeys.TRAIN)
    predicted_output = tf.layers.dense(inputs=dropout, units=4)#x y w h


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicted_output)

    print('Labels x predicted_output',labels,predicted_output)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predicted_output)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predicted_output)}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

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
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=1,
            shuffle=True)

        estimator.train(
            input_fn=train_input_fn,
            steps=None,
            # hooks=[logging_hook]
            )

    # Predict
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        # num_epochs=10,
        shuffle=False)

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
    for pred_bbox, test_bbox in zip(predict_results, eval_labels):
        summed_IOU += IOU(pred_bbox, test_bbox)
    mean_IOU = summed_IOU / len(pred_bboxes)
    print('mean_IOU:',mean_IOU)



if __name__ == "__main__":
    tf.app.run()
