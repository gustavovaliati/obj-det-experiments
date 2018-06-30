import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))

import numpy as np
import argparse, os

from dataset import HelloWorldDataset
from model import *

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint",
    required = False,
    default = None,
    type=str,
    help = "Load tensorflow checkpoint. checkpoint_path: Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.")
ap.add_argument("-md", "--model_dir",
    required = True,
    default = None,
    type=str,
    help = "Dir to save tensorflow checkpoint. Directory to save model parameters, graph and etc. This can also be used to load checkpoints from the directory into a estimator to continue training a previously saved model.")
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
ap.add_argument("-m", "--model_id",
    required=False,
    default=1,
    type=int,
    help = "Defines which model is going to be used.")
ap.add_argument("-v", "--verbose",
    required=False,
    default=False,
    type=bool,
    help = "Enables verbosity")
ap.add_argument("-e", "--epochs",
    required=False,
    default=1,
    type=int,
    help = "Number of epochs.")
ARGS = ap.parse_args()

if ARGS.num_imgs < 30:
    raise Exception('The minimum num_imgs is 30.')

curr_model = None
if ARGS.model_id == 1:
    curr_model = Conv_net_01(img_size=ARGS.img_size, n_classes=ARGS.obj_number)
elif ARGS.model_id == 2:
    curr_model = Conv_net_02(img_size=ARGS.img_size, n_classes=ARGS.obj_number)
elif ARGS.model_id == 3:
    curr_model = Conv_net_03(img_size=ARGS.img_size, n_classes=ARGS.obj_number)
elif ARGS.model_id == 4:
    curr_model = Conv_net_04(img_size=ARGS.img_size, n_classes=ARGS.obj_number)
elif ARGS.model_id == 5:
    curr_model = Conv_net_05(img_size=ARGS.img_size, n_classes=ARGS.obj_number)
else:
    raise Exception('The given model id does not exist.')

tf.logging.set_verbosity(tf.logging.INFO)

print('Using model',curr_model.get_name())

def cnn_model_fn(features, labels, mode):

    logits_train = curr_model.get_model(x=features, reuse=False, is_training=True)
    logits_test = curr_model.get_model(x=features, reuse=True, is_training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=logits_test)
    if ARGS.verbose:
        print('labels.shape,logits_train.shape',labels.shape,logits_train.shape)
    loss_op = tf.losses.mean_squared_error(labels=labels, predictions=logits_train)

    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
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

    # train_data, train_y, test_data, test_y = dataset.generate()
    train_data, train_y, test_data, test_y = dataset.load_or_generate(dataset_path='./datasets')


    if ARGS.verbose:
        # Show the shapes
        print(
            'Shapes: ',
            'train_data', train_data.shape,
            'train_y', train_y.shape,
            'test_data', test_data.shape,
            'test_y', test_y.shape)

    print('Translating gt...')
    new_train_y = translate_to_model_gt(train_y,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True, verbose=ARGS.verbose)
    new_test_y = translate_to_model_gt(test_y,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True, verbose=ARGS.verbose)

    # Show a random sample from the dataset.
    # dataset.show_generated()

    # Create the Estimator
    os.makedirs(ARGS.model_dir, exist_ok=True)
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=os.path.dirname(ARGS.model_dir))

    if not ARGS.checkpoint:
        print("Training...")

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=new_train_y,
            batch_size=64,
            num_epochs=ARGS.epochs,
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
        checkpoint_path=ARGS.checkpoint #checkpoint_path: Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.
        )
    if ARGS.verbose:
        print('eval_results',eval_results)

    # Predict
    print('Predicting...')
    predict_results = estimator.predict(input_fn=eval_input_fn)
    results = np.array(list(predict_results))
    if ARGS.verbose:
        print('results shape',results.shape)

    print('Translating predicitions...')
    pred_translated = translate_from_model_pred(results, curr_model.get_config(),verbose=ARGS.verbose,obj_threshold=0.1)
    print('Executing NMS...')
    pred = do_nms(pred_translated, model_config=curr_model.get_config(), iou_func=dataset.bbox_iou_centered,verbose=ARGS.verbose)
    if ARGS.verbose:
        print('Camera ready predictions (first 10 imgs)...')
        for img_index, img_p in enumerate(pred):
            if img_index >= 10:
                break
            for obj_index, obj_p in enumerate(img_p):
                for obj_gt in test_y[img_index]:
                    iou = dataset.bbox_iou_centered(obj_gt,obj_p)
                    print("For img {} - gt {} <-> pred {} has iou of {}".format(img_index,obj_gt,obj_p,iou))

    mean_iou, iou_per_image = dataset.grv_mean_iou(pred,gt=test_y)
    print('mean_iou: ',mean_iou)

    dataset.show_predicted(predictions=pred,gt=test_y)

def main(unused_argv):
    run_model()

if __name__ == "__main__":
    tf.app.run()
