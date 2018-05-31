from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from jrieke_tf_dataset import JriekeBboxDataset

tf.logging.set_verbosity(tf.logging.INFO)

CHECKPOINT_PATH='/tmp/convnet_model/model.ckpt-80004'

def cnn_model_fn(features, labels, mode):
    # model = Sequential([
    #         Dense(200, input_dim=X.shape[-1]),
    #         Activation('relu'),
    #         Dropout(0.2),
    #         Dense(y.shape[-1])
    #     ])
    # model.compile('adadelta', 'mse')
    input_layer = tf.reshape(features["x"], [-1, 8, 8, 1])
    flat = tf.reshape(input_layer, [-1, 8 * 8 * 1])
    dense = tf.layers.dense(inputs=flat, units=200, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
                inputs=dense,
                rate=0.2,
                training=mode == tf.estimator.ModeKeys.TRAIN)
    predicted_output = tf.layers.dense(inputs=dropout, units=4)#x y w h
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # "classes": tf.argmax(input=predicted_output, axis=1),
        "bboxes": predicted_output,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(predicted_output, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predicted_output)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
    #   optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["bboxes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    print("TensorFlow version: {}".format(tf.__version__))
    dataset = JriekeBboxDataset()
    train_data, train_labels, eval_data, eval_labels = dataset.generate()
    # dataset.show_generated()

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=5000)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # classifier.train(
    #     input_fn=train_input_fn,
    #     steps=20000,
    #     hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(
        input_fn=eval_input_fn,
        checkpoint_path=CHECKPOINT_PATH)
    print(eval_results)


    predict_results = classifier.predict(
        input_fn=eval_input_fn,
        checkpoint_path=CHECKPOINT_PATH)

    # for prediction in predict_results:
        # print(prediction)
    # print(predict_results[0])

    pred_bboxes = [ [r['bboxes']] for r in predict_results]
    dataset.show_predicted(pred_bboxes, eval_labels)


if __name__ == "__main__":
    tf.app.run()
