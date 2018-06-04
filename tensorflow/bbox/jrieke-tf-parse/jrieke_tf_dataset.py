'''
This code is based on https://github.com/jrieke/shape-detection/
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import datetime

class JriekeBboxDataset:
    def generate(self):
        # Create images with random rectangles and bounding boxes.
        num_imgs = 50000

        self.WIDTH = 8
        self.HEIGHT = 8
        min_object_size = 1
        max_object_size = 4
        num_objects = 1

        self.bboxes = np.zeros((num_imgs, num_objects, 4))
        self.imgs = np.zeros((num_imgs, self.WIDTH, self.HEIGHT))  # set background to 0

        for i_img in range(num_imgs):
            for i_object in range(num_objects):
                w, h = np.random.randint(min_object_size, max_object_size, size=2)
                x = np.random.randint(0, self.WIDTH - w)
                y = np.random.randint(0, self.HEIGHT - h)
                self.imgs[i_img, x:x+w, y:y+h] = 1.  # set rectangle to 1
                self.bboxes[i_img, i_object] = [x, y, w, h]

        self.imgs.shape, self.bboxes.shape

        # Reshape and normalize the image data to mean 0 and std 1.
        X = (self.imgs.reshape(num_imgs, -1) - np.mean(self.imgs)) / np.std(self.imgs)
        X.shape, np.mean(X), np.std(X)

        # Normalize x, y, w, h by self.img_size, so that all values are between 0 and 1.
        # Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
        y = self.bboxes.reshape(num_imgs, -1) / self.WIDTH
        y.shape, np.mean(y), np.std(y)

        # Split training and test.
        i = int(0.8 * num_imgs)
        train_X = X[:i]
        test_X = X[i:]
        train_y = y[:i]
        test_y = y[i:]
        self.test_imgs = self.imgs[i:]
        self.test_bboxes = self.bboxes[i:]

        ### check if the generated imgs match to the test_X slice image
        # self.check_dataset_image_compability(test_X[0], self.test_imgs[0])

        return train_X, train_y, test_X, test_y

    def check_dataset_image_compability(self, test_X_sample, test_imgs_sample):
        fig = plt.figure(figsize=(12, 3))
        fig.suptitle('check if the generated imgs match to the test_X slice image')
        fig.subplots_adjust(top=0.85)

        plt.subplot(1, 2, 1)
        plt.gca().set_title('Returned by the dataset class: used for training')
        TMP = test_X_sample.reshape(self.WIDTH, self.HEIGHT)
        plt.imshow(TMP.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])

        plt.subplot(1, 2, 2)
        plt.gca().set_title('Global image holder: used for plotting.')
        plt.imshow(test_imgs_sample.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
        plt.show()
        print('compare:',TMP,test_imgs_sample)

    def IOU(self,bbox1, bbox2):
        '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
        x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

        w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
        h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
        if w_I <= 0 or h_I <= 0:  # no overlap
            return 0.
        I = w_I * h_I

        U = w1 * h1 + w2 * h2 - I

        return I / U

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
        plt.imshow(self.imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
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


    def check_dataset_image_compability(self, test_X_sample, test_imgs_sample):
        fig = plt.figure(figsize=(12, 3))
        fig.suptitle('check if the generated imgs match to the test_X slice image')
        fig.subplots_adjust(top=0.85)

        plt.subplot(1, 2, 1)
        plt.gca().set_title('Returned by the dataset class: used for training')
        plt.imshow(TMP.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])

        plt.subplot(1, 2, 2)
        plt.gca().set_title('Global image holder: used for plotting.')
        plt.imshow(test_imgs_sample.T, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
        plt.show()
        print('compare:',TMP,test_imgs_sample)

    def show_predicted(self, pred_bboxes):
        # Show a few images and predicted bounding boxes from the test dataset.

        fig = plt.figure(figsize=(12, 3))
        fig.subplots_adjust(top=0.85)
        fig.suptitle('Prediction demonstration. Random samples.')
        legend_plotted = False

        for i_subplot in range(1, 11):
            plt.subplot(1, 10, i_subplot)
            i = np.random.randint(len(pred_bboxes))
            plt.imshow(self.test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
            for pred_bbox, exp_bbox in zip(pred_bboxes[i], self.test_bboxes[i]):
                # print('before convertion: pred',pred_bbox, 'gt',exp_bbox)
                pred_bbox = self.convertDefaultAnnotToCoord(pred_bbox)
                # exp_bbox = self.convertDefaultAnnotToCoord(exp_bbox)
                print('after convertion: pred',pred_bbox, 'gt',exp_bbox)
                plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
                #gt
                plt.gca().add_patch(matplotlib.patches.Rectangle((exp_bbox[0], exp_bbox[1]), exp_bbox[2], exp_bbox[3], ec='b', fc='none'))
                plt.annotate('IOU: {:.2f}'.format(self.IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2), color='r')
                if not legend_plotted:
                    legend_plotted = True
                    plt.gca().legend(['Pred','GT'],loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
        plt.show()
        # plt.savefig('plots/bw-single-rectangle_prediction_{0:%Y-%m-%d%H:%M:%S}.png'.format(datetime.datetime.now()), dpi=300)
