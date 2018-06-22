import tensorflow as tf
import numpy as np
import math

#the anchors need to be odd.
# anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326]

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
    def __init__(self, img_size=None, img_channels=3):

        self.img_size = img_size
        self.img_channels = img_channels
        
        '''
        n_grid refers to the number of slices the image is going to be divided,
        generating a number of cells.
        An image of 16x16 with n_grid==4, has 16 cells of 4x4 pixels each

        The n_grid is calculated according to the model architecture.
        Look to the layers outputs to understand the calculation.

        If image is 16x16, then: 16 -> 14 -> 12 -> 5. Final layer output is [5]x5xn_outputs_per_cell
        '''
        self.n_grid = 5
        if self.n_grid < 4:
            raise Exception(' The calculated n_grid value should be at least 4. Seems like the given img_size is too small, try minimum of 16x16')

        self.anchors = [[3,3],[5,5],[7,7]]
        self.n_anchors_per_cell = len(self.anchors)
        self.n_classes = 2
        self.n_bboxes =  self.n_grid*self.n_grid * self.n_anchors_per_cell
        self.n_cells = self.n_grid * self.n_grid

        # if we have only 1 class, we do not need to use the c in the output vector.
        classes = self.n_classes if self.n_classes > 1 else 0
        self.n_outputs_per_anchor = 5 + classes
        self.n_outputs_per_cell = self.n_anchors_per_cell * self.n_outputs_per_anchor
        self.n_outputs = self.n_bboxes * self.n_outputs_per_anchor

    def get_config(self):
        return {
            'n_grid' : self.n_grid, #in how many slices the image is going to be divided
            'n_classes': self.n_classes,
            'anchors' : self.anchors, #the default anchors per cell.
            'img_size' : self.img_size,
            'n_outputs' : self.n_outputs,
            'n_cells' : self.n_cells,
            'n_outputs_per_cell': self.n_outputs_per_cell,
            'n_anchors_per_cell' : self.n_anchors_per_cell,
            'n_bboxes': self.n_bboxes #total number of bboxes/anchors in each image
        }

    def get_model(self, x, reuse, is_training):
        # with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, self.img_size, self.img_size, self.img_channels])
        #lets say the image side size is 16 and we have 3 channels
        # Expected input 16x16x3
        conv1 = tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, activation=tf.nn.relu)
        #input 14x14x32
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
        #input 12x12x64
        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu)
        #input 5x5x128
        conv4 = tf.layers.conv2d(conv3, filters=self.n_outputs_per_cell, kernel_size=1, strides=1, activation=tf.nn.relu)

        #expected output 5x5xn_outputs. We want n_outputs outputs per cell.
        out = tf.contrib.layers.flatten(conv4)

        return out

class Conv_net_03:
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

def translate_to_model_gt(gt, config, iou_func, normalized=False, verbose=False):
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
        - lets call (P x y w h (n_classes * c) ) as 'prediction_unit'

    Output per image
    n_cell *
        n_anchors *
            (P x y w h (n_classes * c)

    '''

    def get_cell_for_gt(gt_bbox, n_grid, img_size):
        '''
        Returns the center of the reponsible cell for the given bbox

        Example:
        img_size = 16
        n_grid = 5

        slice_size = int(15/5 = 3.2) = 3

        In the case of having some space left in the grid, we put all in the last grid line/column.

        '''
        slice_size = int(img_size / n_grid)
        x_center,y_center,w,h,class_label= gt_bbox

        '''
        Lets say x,y are 7,16 and image 16 and n_grid 5

        Slices:        1       2       3       4       5       6
        Pixels:        0-2,    3-5,    6-8,    9-11,   12-14,  15-17

        For x, 7/3 = 2.3
        int(2.3) = 2 = third slice, index 2.

        For y, 16/3 = 5.3
        int(5.3) = 5 = sixth slice = (if slice > n_grid: slice = n_grid-1) = fifth slice, index 4.

        '''
        x_in_slice_index = int(x_center / slice_size)
        if x_in_slice_index+1 > n_grid :
            x_in_slice_index = n_grid-1

        #finds the first pixel of the next slice, and subtracts half of the slice size to find the center of the cell
        cell_x = ((x_in_slice_index+1) * slice_size) - (slice_size/2)

        y_in_slice_index = int(y_center / slice_size)
        if y_in_slice_index+1 > n_grid :
            y_in_slice_index = n_grid-1

        cell_y = ((y_in_slice_index+1) * slice_size) - (slice_size/2)

        cell_index = (x_in_slice_index, y_in_slice_index)

        '''
        if cell_index==(1,2), means the cell is at the second (1) column of cells
        and in the third (2) row.
        If the n_grid is 4, we have 4 colums of cells by 4 rows.
        '''

        # how many full rows do we have? That is y_in_slice_index.
        n_cells = y_in_slice_index * n_grid

        # how many cells do we have in the last row? That is x_in_slice_index+1.
        n_cells = n_cells + (x_in_slice_index+1)

        #Putting our cell matrix in a flatten way, our cell (eg, (1,2)) is the seventh.
        # Four in the first row and 3 in the second (if n_grid is 4)
        # The index starts in zero, so we subtract one.
        cell_index = n_cells-1

        return cell_index, cell_x, cell_y

    def get_anchors_by_cell(cell_center_x, cell_center_y):
        '''
        Get the anchors bboxes in relation to the image according to the
        given x,y cell center.

        The returned anchors are x,y,w,h that should be complemented by the offsets predicted by the network
        '''
        anchors_coords = []
        for anchor in config['anchors']:
            w,h = anchor
            anchors_coords.append([cell_center_x,cell_center_y,w,h])

        return anchors_coords

    def get_best_anchor_for_gt(gt_bbox, n_grid, img_size):

        '''
        For the given bouding box, find in the image which cell and anchor fits
        better to this bbox position.
        That is going to give us the real data needed to compare with out model outputs which is
        usually a single and gigantic vector of values per image. In this vector, each position (value)
        has its meaning/function among anchors, grids, classes etec.
        '''

        cell_index,cell_center_x,cell_center_y = get_cell_for_gt(gt_bbox, n_grid, img_size)

        anchors = get_anchors_by_cell(cell_center_x, cell_center_y)

        iou_scores = []
        for anchor in anchors:
            iou = iou_func(anchor, gt_bbox)
            iou_scores.append(iou)
            if verbose:
                print('for anchor',anchor,'iou',iou)

        best_anchor_index = np.argmax(iou_scores)


        return cell_index, best_anchor_index, anchors[best_anchor_index]

    def get_positions_for_anchorindex(cell_index, anchor_index, n_each_anchor_output, n_anchors_per_cell):
        '''
        Finds the positions in the network output that should receive specific gt values: P,x,y,w,h,c1,...,cN
        '''

        cell_initial_pos = cell_index * (n_each_anchor_output * n_anchors_per_cell)
        anchor_initial_pos = cell_initial_pos + (anchor_index * n_each_anchor_output)
        replacing_positions = np.arange(anchor_initial_pos, anchor_initial_pos + n_each_anchor_output)

        return replacing_positions
    if verbose:
        print('model config: ',config)

    #we already have computed the number o bboxes per image (n_cells * n_anchors_per_cell)
    outputs = np.zeros((len(gt), config['n_outputs']))

    if verbose:
        print('network output size is gonna be: ', outputs.shape)

    for img_index, img_gt in enumerate(gt):
        if verbose:
            print('img_gt',img_gt)
        for obj_gt in img_gt:
            if verbose:
                print('for gt:',obj_gt)

            obj_gt_normalized_backup = []
            if normalized:
                #Reverse from normalized.
                obj_gt_normalized_backup.extend(obj_gt)
                obj_gt[0] = obj_gt[0] * config['img_size']
                obj_gt[1] = obj_gt[1] * config['img_size']
                obj_gt[2] = obj_gt[2] * config['img_size']
                obj_gt[3] = obj_gt[3] * config['img_size']
                if verbose:
                    print('normalized to:', obj_gt)


            cell_index, best_anchor_index, best_anchor = get_best_anchor_for_gt(obj_gt,  config['n_grid'], config['img_size'])
            if verbose:
                print('cell_index, best_anchor_index, best_anchor',cell_index, best_anchor_index,best_anchor)

            cell_center_x, cell_center_y, anchor_w, anchor_h = best_anchor[0],best_anchor[1],best_anchor[2],best_anchor[3]

            #Calculate the offsets.
            #The x,y are relative to the cell.
            top_left_cell_x = cell_center_x - int(config['n_grid']/2)
            top_left_cell_y = cell_center_y - int(config['n_grid']/2)
            if verbose:
                print('top_left',top_left_cell_x,top_left_cell_y)

            gt_x, gt_y, gt_w, gt_h, gt_class = obj_gt[0],obj_gt[1],obj_gt[2],obj_gt[3],int(obj_gt[4])

            offset_x = gt_x - top_left_cell_x
            offset_x = offset_x / config['n_grid'] #normalize in the grid scale
            offset_y = gt_y - top_left_cell_y
            offset_y = offset_y / config['n_grid'] #normalize in the grid scale
            if verbose:
                print('offset_x,offset_y',offset_x,offset_y)

            expoent_w = math.log(gt_w,anchor_w) # gt_w == anchor_w ^ expoent_w
            expoent_h = math.log(gt_h,anchor_h) # gt_h == anchor_h ^ expoent_h
            if verbose:
                print('expoent_w,expoent_h',expoent_w,expoent_h)

            p = 1.0 # Our probability of this bbox having an object.

            c = np.zeros((config['n_classes']),dtype=int)
            c[gt_class] = 1 #activate the gt class

            #gt_output : object_probability, x,y,w,h offsets, class1_probability, ..., classN_probability
            if normalized:
                #Normalized the output
                offset_x = offset_x / config['img_size']
                offset_y = offset_y / config['img_size']
                expoent_w = expoent_w / config['img_size']
                expoent_h = expoent_h / config['img_size']

            gt_output = np.concatenate(([p,offset_x,offset_y,expoent_w,expoent_h],c))
            if verbose:
                print('gt_output',gt_output)

            replacing_positions = get_positions_for_anchorindex(cell_index, best_anchor_index, len(gt_output), len(config['anchors']))

            # We replace a specific location from the entire output corresponding to the gt bbox.
            np.put(outputs[img_index], replacing_positions, gt_output)

            # Now our gigantic output contains the gt values exactly in the anchor it should be related to for training.
            # for value in outputs[img_index]:
            #     print('>',value)

    return outputs

def find_cell_topleft(cell_index,config):
    '''
    How many complete rows we have?
        R: Just the first row is complete, and there are more cells in the next row.
        So we have found the row.

    Lets say we want the anchor_index==4 for a n_grid==3
    [ [1,2,3],[4,(5),6],[7,8,9] ]

    Using divmod(anchor_index+1, n_grid). See some examples:
    >>> divmod(5,3)
    (1, 2) -> our anchor is in the row of index 1 (second row) in the second position (index 1).
    >>> divmod(1,3)
    (0, 1) -> our anchor is in the row of index 0 (first row) in the first position (index 0).
    >>> divmod(9,3)
    (3, 0) ->   Our anchor is in the row of index 3, which gives a n_grid==4.
                Our n_grid is just 3. That means our cell is the last one.
    >>> divmod(8,3)
    (2, 2) -> our anchor is in the row of index 2 (last row) in the second position (index 1).

    '''
    (row_index,rest) = divmod((cell_index + 1), config['n_grid'])
    column_index = rest - 1
    if row_index >= config['n_grid']:
        row_index = config['n_grid'] - 1
        column_index = row_index

    '''
    Lets img_size==16.
    cell_slice_size ==  int(5.333) == 16 / 3 == img_size/n_grid
    The 0.333 remaining we always leave for the last columns/rows
    '''
    cell_slice_size = int(config['img_size'] / config['n_grid'])

    topleft_x = column_index * cell_slice_size
    topleft_y = row_index * cell_slice_size

    # returns normalized
    return topleft_x / config['img_size'], topleft_y / config['img_size']

def calc_scaled_from_offsets(offseted_bboxes, config, anchor_index, cell_index):

    x_offset, y_offset, w_offset, h_offset = offseted_bboxes[0],offseted_bboxes[1],offseted_bboxes[2],offseted_bboxes[3]

    model_anchor = config['anchors'][anchor_index]
    anchor_w, anchor_h = model_anchor[0], model_anchor[1]
    w = math.pow(anchor_w,w_offset)
    h = math.pow(anchor_h,h_offset)

    cell_side = int(config['img_size'] / config['n_grid'])
    #x as pixels relative to the top left corner of the cell
    x_cell_pixels = x_offset * cell_side
    # now find the scale of these pixels to the entire image
    x_image_pixels_scaled = x_cell_pixels * config['img_size']

    topleft_x, topleft_y = find_cell_topleft(cell_index,config)
    x = topleft_x + x_offset
    y = topleft_y + y_offset

    return x,y,w,h

def translate_from_model_pred(pred, config, verbose=False, obj_threshold=0.5):
    '''
    prediction_unit : (P x y w h (n_classes * c) ) -> dimensions contains only offsets
    wanted output is like the gt: x,y,w,h,class -> dimensions are normalized to the image dimensions
    '''
    n_pred = len(pred)
    n_outputs_pred = len(pred)
    if verbose:
        print('Translating predicitions: ', n_pred)
        print('The predictions have outputs of size:', n_outputs_pred)

    pred = np.array(pred)

    translated_output = []

    for img_output in pred:
        translated_img_output = [] #put inside this all relevant predicted bboxes
        cells = np.split(img_output,config['n_cells'])
        for cell_index, cell in enumerate(cells):
            anchors = np.split(cell,config['n_anchors_per_cell'])
            for anchor_index, prediction_unit in enumerate(anchors):
                P, x_offset, y_offset, w_offset, h_offset = prediction_unit[0],prediction_unit[1],prediction_unit[2],prediction_unit[3],prediction_unit[4]
                pred_class = 0
                if len(prediction_unit) > 5:
                    pred_class = np.argmax(prediction_unit[5:-1])

                x,y,w,h = calc_scaled_from_offsets([x_offset, y_offset, w_offset, h_offset],config,anchor_index,cell_index)


                if P >= obj_threshold:
                    print('P',P)
                    translated_img_output.append([x,y,w,h,pred_class])
        translated_output.append(translated_img_output)
    return translated_output
