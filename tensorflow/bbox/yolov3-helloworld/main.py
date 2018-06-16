import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))

import numpy as np
import argparse

from dataset import HelloWorldDataset
from model import translate_to_model_gt, Conv_net_01, Conv_net_02, Conv_net_03

cur_model = Conv_net_02()

tf.logging.set_verbosity(tf.logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint",
    required = False,
    default = None,
    type=str,
    help = "Load tensorflow checkpoint.")
ap.add_argument("-n", "--obj_number",
    required=False,
    default=2,
    type=int,
    help = "Defines how many objects are going to be inserted in each image.")
ap.add_argument("-s", "--shape_number",
    required=False,
    default=2,
    type=int,
    help = "Defines how many object shapes are going to be used.")
ap.add_argument("-i", "--img_size",
    required=False,
    default=8,
    type=int,
    help = "Defines the value for image's width and height. It is going to be a square.")
ARGS = ap.parse_args()

def cnn_model_fn(features, labels, mode):

    logits_train = cur_model.get_model(x=features, reuse=False, is_training=True)
    logits_test = cur_model.get_model(x=features, reuse=True, is_training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=logits_test)

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

def run_model():
    dataset = HelloWorldDataset(
        num_objects=ARGS.obj_number,
        shape_number=ARGS.shape_number,
        img_size=ARGS.img_size,
        train_proportion=0.8)

    train_data, train_y, test_data, test_y = dataset.generate()
    # Show the shapes
    print(
        'Shapes: ',
        'train_data', train_data.shape,
        'train_y', train_y.shape,
        'test_data', test_data.shape,
        'test_y', test_y.shape)

    train_y = translate_to_model_gt(train_y, cur_model.get_config(img_size=ARGS.img_size))
    test_y = translate_to_model_gt(test_y, cur_model.get_config(img_size=ARGS.img_size))

    # Show a random sample from the dataset.
    dataset.show_generated()

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

    if not ARGS.checkpoint:
        print("Training...")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_y,
            batch_size=64,
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
        y=test_y,
        # num_epochs=10,
        shuffle=False)

    eval_results = estimator.evaluate(
        input_fn=eval_input_fn,
        checkpoint_path=ARGS.checkpoint
        )
    print('eval_results',eval_results)

    # Predict
    predict_results = estimator.predict(input_fn=eval_input_fn)
    results = np.array(list(predict_results))
    print('results shape',results.shape)


def main(unused_argv):
    run_model()


if __name__ == "__main__":
    tf.app.run()