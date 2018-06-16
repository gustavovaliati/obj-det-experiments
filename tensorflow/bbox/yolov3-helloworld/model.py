import tensorflow as tf

#anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

class Conv_net_01:
    def calc_grid(img_size=None):
        return (img_size)/2 - 2 - 2
    def get_model(x, reuse, is_training, n_outputs):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = tf.reshape(x, shape=[-1, args.img_size, args.img_size, 1]) # img channels == 1
            #lets say the image side size is 16
            # Expected input 16x16x1
            #Downsample
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=2, activation=tf.nn.relu)
            #input 8x8x32
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            #input 6x6x64
            conv3 = tf.layers.conv2d(conv3, filters=128, kernel_size=3, strides=1, activation=tf.nn.relu)
            #input 4x4x128
            conv4 = tf.layers.conv2d(conv3, filters=n_outputs, kernel_size=1, strides=1, activation=tf.nn.relu)
            #expected output 4x4xn_outputs. We want n_outputs outputs per cell.
            out = tf.contrib.layers.flatten(conv4)
        return out

class Conv_net_02:
    def get_config(self,img_size):
        n_grid = self.calc_grid(img_size)
        anchors = [[1,1], [2,2], [3,3]]
        return {
            'n_grid' : ,
            'n_classes': 2,
            'anchors' : anchors,
            'img_size' : img_size,
            'n_bboxes': n_grid*n_grid * len(anchors)
        }
    def calc_grid(img_size=None):
        return (img_size - 2 - 2) /2 -2
    def get_model(x, reuse, is_training,n_outputs):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = tf.reshape(x, shape=[-1, args.img_size, args.img_size, 1]) # img channels == 1
            #lets say the image side size is 16
            # Expected input 16x16x1
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, activation=tf.nn.relu)
            #input 14x14x32
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
            #input 12x12x64
            conv3 = tf.layers.conv2d(conv3, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu)
            #input 6x6x128
            conv4 = tf.layers.conv2d(conv3, filters=n_outputs, kernel_size=1, strides=1, activation=tf.nn.relu)

            #expected output 4x4xn_outputs. We want n_outputs outputs per cell.
            out = tf.contrib.layers.flatten(conv4)

        return out

class Conv_net_03(x, reuse, is_training):
    def calc_grid(self,img_size=None):
        return (img_size - 2 - 2)/2 -2
    def get_model(self,x, reuse, is_training, n_outputs):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = tf.reshape(x, shape=[-1, args.img_size, args.img_size, 1]) # img channels == 1
            #lets say the image side size is 16
            #input 16x16x1
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, activation=tf.nn.relu)

            #input 14x14x32
            conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            #input 12x12x64
            #Downsample
            conv3 = tf.layers.conv2d(conv3, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu)

            #input 6x6x128
            conv4 = tf.layers.conv2d(conv1, filters=256, kernel_size=3, strides=1, activation=tf.nn.relu)

            #input 4x4x128
            conv5 = tf.layers.conv2d(conv3, filters=n_outputs, kernel_size=1, strides=1, activation=tf.nn.relu)

            #expected output 4x4xn_outputs. We want n_outputs outputs per cell.
            out = tf.contrib.layers.flatten(conv5)

        return out

def translate_to_model_gt(gt, config):
    '''
    Receives gt bbox coordinates for the objects for each image and
    organize them according to the expected model output, so that
    it is possible to match both for loss application.

    Expected as gt something like:
    image_quantity * object_quantity * (x,y,w,h)

    Will translate in:
    image_quantity * (n_grid * n_grid) * n_anchors_per_cell *  (P x y w h (n_classes * c) )

    Ps:
        - If n_classes == 1, we will remove the (n_classes * c) since it is unnecessary.
        - anchors == None, means to use them all.
        - (n_grid * n_grid) = number of cells per image, depends on the image dimension and net structure.

    '''

    def get_cell_for_gt(gt_bbox, n_grid, img_size):
        '''
        Returns the center of the reponsible cell for the given bbox
        '''

        return cell_index,x,y

    def get_anchors_by_cell(cell_center_x, cell_center_y):
        '''
        Get the anchors bboxes in relation to the image according to the
        given x,y cell center.
        '''

        return [anchor, anchor]


    def get_best_fit_for_gt(gt_bbox,  n_grid, img_size):

        '''
        For the given bouding box, find in the image which cell and anchor fits
        better to this bbox position.
        That is going to give us the real data needed to compare with out model outputs
        '''

        cell_index,cell_center_x,cell_center_y = get_cell_for_gt(gt_bbox, n_grid, img_size)

        anchors = get_anchors_by_cell(cell_center_x, cell_center_y)

        iou_scores = []
        for anchor in anchors:
            iou_scores.append(database.iou_centered(anchor, gt_bbox))

        best_anchor_index = np.argmax(iou_scores)


        return cell_index, anchor_index

    '''
    Output per image
    n_cell *
        n_anchors *
            (P x y w h (n_classes * c)

    '''

    output = np.zeros((len(gt), () ))
    for img_bboxes in gt:
        for gt_obj_bbox in img_bboxes:
            cell_index, anchor_index = get_best_anchor_for_gt(gt_obj_bbox,  config['n_grid'], config['img_size'])
