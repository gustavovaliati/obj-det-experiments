'''
This code is based on https://github.com/jrieke/shape-detection/
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime

class HelloWorldDataset:
    def __init__(self, num_imgs = 50000, min_object_size = 1, max_object_size = 4,
        num_objects = 1, img_size = 8, train_proportion = 0.8,
        shape_number=2):

        self.num_imgs = num_imgs
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size
        self.num_objects = num_objects
        self.WIDTH = img_size #For now this should work only for square.
        self.HEIGHT = img_size
        self.train_proportion = train_proportion

        if shape_number > 2:
            raise Exception("We actually support only a maximum of 3 shapes.")
        self.shape_number = shape_number
        self.shape_labels = ['rectangle','triangle','circle']
        self.shape_labels_colors = ['g','y','c']

    def generate(self):
        print('Generating...')

        self.bboxes = np.zeros((self.num_imgs, self.num_objects, 4))
        self.imgs = np.zeros((self.num_imgs, self.WIDTH, self.HEIGHT))  # set background to 0
        self.shapes = np.zeros((self.num_imgs, self.num_objects), dtype=int)
        print('self.shapes.shape = ',self.shapes.shape)

        for i_img in range(self.num_imgs):
            for i_object in range(self.num_objects):
                shape = np.random.randint(self.shape_number)
                self.shapes[i_img, i_object] = shape
                if shape == 0:
                    w, h = np.random.randint(self.min_object_size, self.max_object_size, size=2)
                    x = np.random.randint(0, self.WIDTH - w)
                    y = np.random.randint(0, self.HEIGHT - h)
                    self.imgs[i_img, y:y+h, x:x+w] = 1.  # set rectangle to 1
                    self.bboxes[i_img, i_object] = [x, y, w, h]
                elif shape == 1:
                    size = np.random.randint(self.min_object_size, self.max_object_size)
                    x = np.random.randint(0, self.WIDTH - size)
                    y = np.random.randint(0, self.HEIGHT - size)
                    mask = np.tril_indices(size)
                    self.imgs[i_img, y + mask[0], x + mask[1]] = 1.
                    self.bboxes[i_img, i_object] = [x, y, size, size]
                else:
                    raise Exception("Unsupported requested shape quantity.")

        print("Shapes: imgs ", self.imgs.shape, " bboxes ", self.bboxes.shape)

        #why this?
        # X = (self.imgs.reshape(self.num_imgs, -1) - np.mean(self.imgs)) / np.std(self.imgs)
        X = self.imgs


        shapes_onehot = np.zeros((self.num_imgs, self.num_objects, self.shape_number))
        for i_img in range(self.num_imgs):
            for i_object in range(self.num_objects):
                shapes_onehot[i_img, i_object, self.shapes[i_img, i_object]] = 1

        # y = self.bboxes.reshape(self.num_imgs, -1) / self.WIDTH
        y = np.concatenate( [self.bboxes / self.WIDTH, shapes_onehot], axis=-1).reshape(self.num_imgs, -1)
        print('y shape', y.shape)
        # Split training and test.
        i = int(self.train_proportion * self.num_imgs)
        train_X = X[:i] #80% for training
        test_X = X[i:]
        train_y = y[:i]
        test_y = y[i:]
        self.test_imgs = self.imgs[i:]
        self.test_bboxes = self.bboxes[i:]

        return train_X, train_y, test_X, test_y

    def bbox_iou(self,boxA, boxB):
        #From: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

        raise Exception('TODO, check if we are dealing with x and y being center of bbox or top-left')
        # A) x1,y2,w,h
        A_x1, A_y1, A_w, A_h = boxA[0], boxA[1], boxA[2], boxA[3]
        # A) x2,y2
        A_x2, A_y2 = A_x1 + A_w, A_y1 + A_h

        # B) x1,y2,w,h
        B_x1, B_y1, B_w, B_h = boxB[0], boxB[1], boxB[2], boxB[3]
        # B) x2,y2
        B_x2, B_y2 = B_x1 + B_w, B_y1 + B_h

        xA = max(A_x1, B_x1)
        yA = max(A_y1, B_y1)
        xB = min(A_x2, B_x2)
        yB = min(A_y2, B_y2)

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (A_x2 - A_x1 + 1) * (A_y2 - A_y1 + 1)
        boxBArea = (B_x2 - B_x1 + 1) * (B_y2 - B_y1 + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def convertDefaultAnnotToCoord(self, annot):
        '''
        annot -> [x, y, w, h]
        '''

        w = annot[2] * self.WIDTH
        h = annot[3] * self.HEIGHT

        x = annot[0] * self.HEIGHT
        y = annot[1] * self.HEIGHT

        return [x,y,w,h]

    def convertYoloAnnotToCoord(self, yolo_annot):
        '''
        yolo_annot -> [x, y, w, h]
        '''

        w = yolo_annot[2] * self.WIDTH
        h = yolo_annot[3] * self.HEIGHT

        x = (yolo_annot[0] * self.WIDTH) - (w/2)
        y = (yolo_annot[1] * self.HEIGHT) - (h/2)

        return [x,y,w,h]

    def show_generated(self, i=0):
        fig = plt.figure()
        fig.subplots_adjust(top=0.85)
        fig.suptitle('Generated image sample + GT')
        plt.imshow(self.imgs[i], cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
        for bbox in self.bboxes[i]:
            plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
            plt.gca().legend(['GT'])
        plt.show()

    def plot_rectangle(self, img, bbox):

        fig = plt.figure()
        fig.suptitle('Plotting rectangle.')
        fig.subplots_adjust(top=0.85)

        plt.subplot(1, 1, 1)
        plt.imshow(img, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
        plt.show()

    def show_predicted(self, pred_bboxes, pred_shapes_onehot):
        # Show a few images and predicted bounding boxes from the test dataset.

        fig = plt.figure(figsize=(12, 3))
        fig.subplots_adjust(top=0.85)
        fig.suptitle('Prediction demonstration. Random samples.')
        legend_plotted = False

        for i_subplot in range(1, 6):
            plt.subplot(1, 5, i_subplot)
            i = np.random.randint(len(pred_bboxes))
            plt.imshow(self.test_imgs[i], cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
            for pred_bbox, exp_bbox, pred_shape_idx in zip(pred_bboxes[i], self.test_bboxes[i], pred_shapes_onehot[i]):
                # print('before convertion: pred',pred_bbox, 'gt',exp_bbox)
                pred_bbox = self.convertDefaultAnnotToCoord(pred_bbox)
                # exp_bbox = self.convertDefaultAnnotToCoord(exp_bbox)
                # print('after convertion: pred',pred_bbox, 'gt',exp_bbox)
                plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
                #gt
                plt.gca().add_patch(matplotlib.patches.Rectangle((exp_bbox[0], exp_bbox[1]), exp_bbox[2], exp_bbox[3], ec='b', fc='none'))

                plt.annotate(
                    'IOU:{:.2f},{}'.format(self.IOU(pred_bbox, exp_bbox),self.shape_labels[pred_shape_idx]),
                    (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2),
                    color=self.shape_labels_colors[pred_shape_idx])
                if not legend_plotted:
                    legend_plotted = True
                    plt.gca().legend(['Pred','GT'],loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
        plt.show()
        # plt.savefig('plots/bw-single-rectangle_prediction_{0:%Y-%m-%d%H:%M:%S}.png'.format(datetime.datetime.now()), dpi=300)
