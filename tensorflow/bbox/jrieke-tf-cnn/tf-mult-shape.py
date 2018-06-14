from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse

#from jrieke_tf_dataset import JriekeBboxDataset
from jrieke_tf_dataset_multishape import JriekeBboxDataset
from metrics import Metrics

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):

    def conv_net_02(x, n_outputs, dropout, reuse, is_training):
        with tf.variable_scope('ConvNet', reuse=reuse):
            # cnn from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

            '''
            Removed the max_pooling2d layeres.
            mean_IOU increased from 0.7805 to 0.8731. But got almost 10 times slower to train.
            '''
            x = tf.reshape(x, shape=[-1, args.img_size, args.img_size, 1])
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

            x = tf.reshape(x, shape=[-1, args.img_size, args.img_size, 1])
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
        '''
        Outputs
        4 coords: x,y,w,h
        onehot vector for each object. If there are three shapes, the vector length is 3: [shape_1, shape_2, shape_3]
        '''
        print('CNN outputs number: ', output_number)

        logits_train = conv_net_01(x=features, n_outputs=output_number, dropout=0.25, reuse=False, is_training=True)
        logits_test = conv_net_01(x=features, n_outputs=output_number, dropout=0.25, reuse=True, is_training=False)

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

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint",
                    required = False,
                    default = False,
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
    args = ap.parse_args()

    CHECKPOINT_PATH = None
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint

    print("TensorFlow version: {}".format(tf.__version__))

    dataset = JriekeBboxDataset(num_objects=args.obj_number, shape_number=args.shape_number, img_size=args.img_size, train_proportion=0.9999)
    train_data, train_y, test_data, test_y = dataset.generate()
    output_number = args.obj_number * (4 +  args.shape_number)

    # Show the shapes
    print(
        'Shapes: ',
        'train_data', train_data.shape,
        'train_y', train_y.shape,
        'test_data', test_data.shape,
        'test_y', test_y.shape)


    # Show a random sample from the dataset.
    dataset.show_generated()

    # Show random samples from the returned sets.

    # rand_index = np.random.randint(0, len(train_data)-1)
    # dataset.plot_rectangle(
    #     train_data[rand_index],
    #     dataset.convertDefaultAnnotToCoord(train_y[rand_index])
    # )
    #
    # rand_index = np.random.randint(0, len(test_data)-1)
    # dataset.plot_rectangle(
    #     test_data[rand_index],
    #     dataset.convertDefaultAnnotToCoord(test_y[rand_index])
    # )

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

    if not CHECKPOINT_PATH:
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
    results = np.array(list(predict_results))
    print('results original shape',results.shape)

    results = results.reshape(-1,args.obj_number, int(output_number / args.obj_number) ) # number of images, number of bboxes per image, number of coords
    print('results shape',results.shape)
    pred_bboxes = results[:,:,:4] #get just the coords
    print('pred_bboxes new shape',pred_bboxes.shape)
    print('pred_bboxes output sample', pred_bboxes[0])
    pred_shapes = results[:,:,4:] #get the shapes
    print('pred_shapes new shape',pred_shapes.shape)
    print('pred_shapes output sample', pred_shapes[0])
    pred_shapes_list = np.argmax(pred_shapes, axis=-1).astype(int)
    print('pred_shapes_list shape',pred_shapes_list.shape)
    print('pred_shapes_list output sample', pred_shapes_list[0])
    print('test_y',test_y.shape)

    test_y_reshaped = test_y.reshape(-1,args.obj_number, int(output_number / args.obj_number) )
    test_bboxes = test_y_reshaped[:,:,:4] #get just the coords
    test_shapes = test_y_reshaped[:,:,4:]
    test_shapes_list = np.argmax(test_shapes, axis=-1).astype(int) #get the shapes

    acc_shapes = np.mean(np.argmax(pred_shapes_list, axis=-1) == np.argmax(test_shapes_list, axis=-1))
    print('acc_shapes',acc_shapes)

    dataset.show_predicted(pred_bboxes, pred_shapes_list)
    #
    # '''
    # Keras:
    #  - 6s - loss: 0.0248 - val_loss: 6.0943e-04
    # '''
    #
    # summed_IOU = 0.
    # for pred_bbox, test_bbox in zip(pred_bboxes, test_y):
    #     # print(pred_bbox[0], test_bbox)
    #     summed_IOU += dataset.IOU(pred_bbox[0], test_bbox)
    # mean_IOU = summed_IOU / len(pred_bboxes)
    # print('mean_IOU:',mean_IOU)
    #
    # pred_labels = np.full((pred_bboxes.shape[0],args.obj_number),0)
    # pred_scores = np.full((pred_bboxes.shape[0],args.obj_number),0)
    # gt_bboxes = test_y.reshape(-1,args.obj_number,4)
    # gt_labels = pred_labels
    # print(pred_bboxes.shape,pred_labels.shape,pred_scores.shape,gt_bboxes.shape,gt_labels.shape)
    # data = (pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels)
    # metrics = Metrics(data)
    # metrics.calc()



if __name__ == "__main__":
    tf.app.run()
