import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))

import numpy as np
import argparse

from dataset import HelloWorldDataset
from model import translate_from_model_pred, translate_to_model_gt, Conv_net_01, Conv_net_02, Conv_net_03

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
    default=16,
    type=int,
    help = "Defines the value for image's width and height. It is going to be a square.")
ap.add_argument("-di", "--num_imgs",
    required=False,
    default=100,
    type=int,
    help = "Defines the number of images generated for the dataset.")
ARGS = ap.parse_args()

if ARGS.num_imgs < 30:
    raise Exception('The minimum num_imgs is 30.')

curr_model = Conv_net_02(img_size=ARGS.img_size, n_classes=2)


def cnn_model_fn(features, labels, mode):

    logits_train = curr_model.get_model(x=features, reuse=False, is_training=True)
    logits_test = curr_model.get_model(x=features, reuse=True, is_training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=logits_test)
    print('labels.shape,logits_train.shape',labels.shape,logits_train.shape)
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
        # num_imgs=100,
        num_objects=ARGS.obj_number,
        shape_number=ARGS.shape_number,
        img_size=ARGS.img_size,
        train_proportion=0.8,
        num_imgs=ARGS.num_imgs)

    train_data, train_y, test_data, test_y = dataset.generate()

    # Show the shapes
    print(
        'Shapes: ',
        'train_data', train_data.shape,
        'train_y', train_y.shape,
        'test_data', test_data.shape,
        'test_y', test_y.shape)

    print('Translating gt...')
    new_train_y = translate_to_model_gt(train_y,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True)
    new_test_y = translate_to_model_gt(test_y,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True, verbose=True)
    print('Done.')

    # Show a random sample from the dataset.
    dataset.show_generated()

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

    if not ARGS.checkpoint:
        print("Training...")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=new_train_y,
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
        y=new_test_y,
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

    pred = translate_from_model_pred(new_test_y, curr_model.get_config(),verbose=True,obj_threshold=0.01)
    # pred = translate_from_model_pred(results, curr_model.get_config(),verbose=True,obj_threshold=0.01)
    for p in pred:
        print('p',p)

    mean_iou, iou_per_image = dataset.grv_mean_iou(pred,gt=test_y)
    print('mean_iou',mean_iou)

    dataset.show_predicted(predictions=pred,gt=test_y)

def main(unused_argv):
    run_model()


if __name__ == "__main__":
    tf.app.run()
