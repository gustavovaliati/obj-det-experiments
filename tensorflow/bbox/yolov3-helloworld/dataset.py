'''
This code is inspired on https://github.com/jrieke/shape-detection/
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import random
# import cairo,math
from skimage.draw import circle, polygon
from tqdm import tqdm


class HelloWorldDataset:
    def __init__(self, num_imgs = 50000, min_object_size = 2, max_object_size = 10,
        num_objects = 1, img_size = 16, train_proportion = 0.8,
        shape_number=2):

        self.num_imgs = num_imgs
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size
        self.num_objects = num_objects

        self.WIDTH = img_size #For now this should work only for square.
        self.HEIGHT = img_size
        self.img_size = img_size
        if not (self.img_size == self.WIDTH == self.HEIGHT):
            raise Exception('For now, we support only squared images.')

        self.train_proportion = train_proportion
        self.test_bboxes = None

        if shape_number > 3:
            raise Exception("For now, we support only a maximum of 3 shapes.")
        self.shape_number = shape_number
        self.shape_labels = ['rectangle','circle','triangle']
        self.shape_labels_colors = ['g','y','c']

    def generate(self):
        print('Generating the dataset...')
        self.y = np.zeros((self.num_imgs, self.num_objects, 5)) #one for the class
        self.imgs = np.zeros((self.num_imgs, self.WIDTH, self.HEIGHT),dtype=np.double)

        for i_img in tqdm(range(self.num_imgs)):

            has_overlap = True
            #Through brute force we are going to generate only objects with low overlap.
            while has_overlap:
                #reset data
                self.y[i_img,:,:] = .0
                self.imgs[i_img,:,:] = .0

                #TODO : randomize the number of objects in each image

                for i_object in range(self.num_objects):
                    shape = np.random.randint(self.shape_number)
                    if shape == 0:  # rectangle
                        w, h = np.random.randint(self.min_object_size, self.max_object_size, size=2)
                        x = np.random.randint(0, self.img_size - w)
                        y = np.random.randint(0, self.img_size - h)
                        rect_vertices = np.array((
                            (y,x),
                            (y,x+w),
                            (y+h,x+w),
                            (y+h,x),
                            (y,x)
                        ))
                        rr, cc = polygon(rect_vertices[:, 0], rect_vertices[:, 1],(self.img_size,self.img_size))
                        yolo_bbox = self.from_default_to_yolo([x, y, w, h])
                        self.validate_bbox(yolo_bbox,annot_type='yolo')
                        self.y[i_img, i_object] = np.concatenate((yolo_bbox,[shape]))
                        self.imgs[i_img,rr,cc] = 1

                    elif shape == 1:  # circle
                        d = np.random.randint(8, self.max_object_size) #diameter
                        r = int(0.5 * d) # radius
                        x = np.random.randint(r, self.img_size - d)
                        y = np.random.randint(r, self.img_size - d)
                        w = d
                        h = d
                        yolo_bbox = self.from_default_to_yolo([x, y, w, h])
                        self.validate_bbox(yolo_bbox,annot_type='yolo')
                        denormalized_yolo_bbox = self.denormalize(yolo_bbox)
                        rr, cc = circle(denormalized_yolo_bbox[1],denormalized_yolo_bbox[0], r,(self.img_size,self.img_size))
                        self.y[i_img, i_object] = np.concatenate((yolo_bbox,[shape]))
                        self.imgs[i_img,rr,cc] = 1
                    elif shape == 2:  # triangle
                        size = np.random.randint(3, self.max_object_size)
                        x = np.random.randint(0, self.img_size - size)
                        y = np.random.randint(0, self.img_size - size)
                        triangle_vertices = np.array((
                            (y,x),
                            (y,x+size),
                            (y+size, x),
                            (y,x)
                        ))
                        rr, cc = polygon(triangle_vertices[:, 0], triangle_vertices[:, 1],(self.img_size,self.img_size))
                        yolo_bbox = self.from_default_to_yolo([x, y, size, size])
                        self.validate_bbox(yolo_bbox,annot_type='yolo')
                        self.y[i_img, i_object] = np.concatenate((yolo_bbox,[shape]))
                        self.imgs[i_img,rr,cc] = 1

                accumulated_iou = 0
                for i_object_compare in range(self.num_objects):
                    for i_object_other in range(self.num_objects):
                        if i_object_other == i_object_compare:
                            #do not compare the same object.
                            continue
                        accumulated_iou += self.bbox_iou_centered(
                            self.denormalize(self.y[i_img][i_object_compare]),
                            self.denormalize(self.y[i_img][i_object_other]))
                has_overlap = True if accumulated_iou > 0.0 else False

        print("Shapes: imgs ", self.imgs.shape)
        print('Dataset: y shape', self.y.shape)

        # Split training and test.
        i = int(self.train_proportion * self.num_imgs)
        train_X = self.imgs[:i] #80% for training
        test_X = self.imgs[i:]
        train_y = self.y[:i]
        test_y = self.y[i:]
        self.test_imgs = self.imgs[i:]
        self.test_bboxes = self.y[i:]

        # print('inside the generated',self.y[0],test_y[0],train_y[0])

        return train_X, train_y, test_X, test_y

    def get_dataset_name(self):
        return "dataset_{}{}{}{}{}{}{}".format(self.num_imgs,self.min_object_size,self.max_object_size,self.num_objects,self.img_size,self.train_proportion,self.shape_number)

    def generate_cairo(self):
        raise Exception('This generates images with dtype=np.uint8 which is incompatible with tensorflow operations')
        self.y = np.zeros((self.num_imgs, self.num_objects, 5)) #one for the class
        self.imgs = np.zeros((self.num_imgs, self.WIDTH, self.HEIGHT,4),dtype=np.uint8)

        for i_img in range(self.num_imgs):

            has_overlap = True
            #Through brute force we are going to generate only objects with low overlap.
            while has_overlap:

                surface = cairo.ImageSurface.create_for_data(self.imgs[i_img], cairo.FORMAT_ARGB32, self.img_size, self.img_size)
                cr = cairo.Context(surface)

                # Fill background white.
                cr.set_source_rgb(1, 1, 1)
                cr.paint()

                #TODO : randomize the number of objects in each image
                # for i_object in range(np.random.randint(self.num_objects)+1):

                for i_object in range(self.num_objects):
                    shape = np.random.randint(self.shape_number)
                    if shape == 0:  # rectangle
                        w, h = np.random.randint(self.min_object_size, self.max_object_size, size=2)
                        x = np.random.randint(0, self.img_size - w)
                        y = np.random.randint(0, self.img_size - h)
                        cr.rectangle(x, y, w, h)
                        yolo_bbox = self.from_default_to_yolo([x, y, w, h])
                        self.validate_bbox(yolo_bbox,annot_type='yolo')
                        self.y[i_img, i_object] = np.concatenate((yolo_bbox,[shape]))
                    elif shape == 1:  # circle
                        r = int(0.5 * np.random.randint(4, self.max_object_size))
                        x = np.random.randint(r, self.img_size - r)
                        y = np.random.randint(r, self.img_size - r)
                        cr.arc(x, y, r, 0, 2*np.pi)
                        x = x - r
                        y = y - r
                        w = 2 * r
                        h = w
                        yolo_bbox = self.from_default_to_yolo([x, y, w, h])
                        self.validate_bbox(yolo_bbox,annot_type='yolo')
                        self.y[i_img, i_object] = np.concatenate((yolo_bbox,[shape]))
                    elif shape == 2:  # triangle
                        size = np.random.randint(3, self.max_object_size)
                        x = np.random.randint(0, self.img_size - size)
                        y = np.random.randint(0, self.img_size - size)
                        cr.move_to(x, y)
                        cr.line_to(x+size, y)
                        cr.line_to(x+size, y+size)
                        cr.line_to(x, y)
                        cr.close_path()
                        yolo_bbox = self.from_default_to_yolo([x, y, size, size])
                        self.validate_bbox(yolo_bbox,annot_type='yolo')
                        self.y[i_img, i_object] = np.concatenate((yolo_bbox,[shape]))

                cr.set_source_rgb(0,0,0)
                cr.fill()

                accumulated_iou = 0
                for i_object_compare in range(self.num_objects):
                    for i_object_other in range(self.num_objects):
                        if i_object_other == i_object_compare:
                            #do not compare the same object.
                            continue
                        accumulated_iou += self.bbox_iou_centered(self.y[i_img][i_object_compare],self.y[i_img][i_object_other])

                has_overlap = True if accumulated_iou > 0.2 else False

        self.imgs = self.imgs[..., 2::-1] # change to RGB

        print("Shapes: imgs ", self.imgs.shape)
        print('Dataset: y shape', self.y.shape)

        # Split training and test.
        i = int(self.train_proportion * self.num_imgs)
        train_X = self.imgs[:i] #80% for training
        test_X = self.imgs[i:]
        train_y = self.y[:i]
        test_y = self.y[i:]
        self.test_imgs = self.imgs[i:]
        self.test_bboxes = self.y[i:]

        print('inside the generated',self.y[0],test_y[0],train_y[0])

        return train_X, train_y, test_X, test_y

    def generate_old(self):
        print('Generating...')

        self.y = np.zeros((self.num_imgs, self.num_objects, 5)) #one for the class
        self.imgs = np.zeros((self.num_imgs, self.WIDTH, self.HEIGHT))  # set background to 0
        # self.shapes = np.zeros((self.num_imgs, self.num_objects), dtype=int)

        for i_img in range(self.num_imgs):
            for i_object in range(self.num_objects):
                shape = np.random.randint(self.shape_number)
                if shape == 0:
                    w, h = np.random.randint(self.min_object_size, self.max_object_size, size=2)

                    x = np.random.randint(0, self.WIDTH - w)
                    y = np.random.randint(0, self.HEIGHT - h)
                    self.imgs[i_img, y:y+h, x:x+w] = 1.  # set rectangle to 1
                    x_center = x + int(w/2)
                    y_center = y + int(h/2)
                    coords = np.array([x_center, y_center, w, h]) / self.img_size
                    print('->',[x_center, y_center, w, h],coords)
                    self.y[i_img, i_object] = np.concatenate((coords,[shape]))
                elif shape == 1:
                    size = np.random.randint(self.min_object_size, self.max_object_size)
                    x = np.random.randint(0, self.WIDTH - size)
                    y = np.random.randint(0, self.HEIGHT - size)
                    mask = np.tril_indices(size)
                    self.imgs[i_img, y + mask[0], x + mask[1]] = 1.
                    x_center = x + int(size/2)
                    y_center = y + int(size/2)
                    coords = np.array([x_center, y_center, size, size]) / self.img_size
                    self.y[i_img, i_object] = np.concatenate((coords,[shape]))
                else:
                    raise Exception("Unsupported requested shape quantity.")

        print("Shapes: imgs ", self.imgs.shape, " bboxes ", self.y.shape)
        print('Dataset: y shape', self.y.shape)

        # Split training and test.
        i = int(self.train_proportion * self.num_imgs)
        train_X = self.imgs[:i] #80% for training
        test_X = self.imgs[i:]
        train_y = self.y[:i]
        test_y = self.y[i:]
        self.test_imgs = self.imgs[i:]
        self.test_bboxes = self.y[i:]

        print('inside the generated',self.y[0],test_y[0],train_y[0])

        return train_X, train_y, test_X, test_y

    def bbox_iou_centered(self,boxA,boxB):
        A_x1, A_y1, A_w, A_h = boxA[0], boxA[1], boxA[2], boxA[3]
        A_x1 = A_x1 - int(A_w/2)
        A_y1 = A_y1 - int(A_h/2)

        B_x1, B_y1, B_w, B_h = boxB[0], boxB[1], boxB[2], boxB[3]
        B_x1 = B_x1 - int(B_w/2)
        B_y1 = B_y1 - int(B_h/2)

        # print(A_x1,A_y1)

        return self.bbox_iou([A_x1, A_y1, A_w, A_h],[B_x1, B_y1, B_w, B_h])

    def bbox_iou(self,boxA, boxB):
        #From: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

        # A) x1,y2,w,h
        A_x1, A_y1, A_w, A_h = boxA[0], boxA[1], boxA[2], boxA[3]
        # A) x2,y2
        A_x2, A_y2 = A_x1 + A_w - 1, A_y1 + A_h - 1

        # B) x1,y2,w,h
        B_x1, B_y1, B_w, B_h = boxB[0], boxB[1], boxB[2], boxB[3]
        # B) x2,y2
        B_x2, B_y2 = B_x1 + B_w - 1, B_y1 + B_h - 1

        xA = max(A_x1, B_x1)
        yA = max(A_y1, B_y1)
        xB = min(A_x2, B_x2)
        yB = min(A_y2, B_y2)

        interArea = max(0,(xB - xA + 1)) * max(0,(yB - yA + 1))

        boxAArea = (A_x2 - A_x1 + 1) * (A_y2 - A_y1 + 1)
        boxBArea = (B_x2 - B_x1 + 1) * (B_y2 - B_y1 + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def from_default_to_yolo(self,annot):
        '''
        from
            not normalized : topleft_x,topleft_y,width,height
        to
            normalized : center_x,center_y,width,height
        '''
        topleft_x,topleft_y,width,height = annot[0],annot[1],annot[2],annot[3]
        # print('topleft_x,topleft_y,width,height',topleft_x,topleft_y,width,height)

        center_x = topleft_x + int(width/2)
        center_y = topleft_y + int(height/2)

        return np.array([center_x,center_y,width,height]) / self.img_size # normalize

    def from_yolo_to_default(self,annot):
        '''
        from
            normalized : center_x,center_y,width,height
        to
            not normalized : topleft_x,topleft_y,width,height
        '''
        # Be aware the class (annot[4]) has been messed up with this denormalization.
        annot = np.multiply(annot, self.img_size) #denormalize
        center_x,center_y,width,height = annot[0],annot[1],annot[2],annot[3]
        # print('center_x,center_y,width,height',center_x,center_y,width,height)

        topleft_x = center_x - int(width/2)
        topleft_y = center_y - int(height/2)

        return [topleft_x,topleft_y,width,height]

    def denormalize(self,annot):
        if len(annot) == 5:
            bkp_class = annot[4]
            annot = np.multiply(annot, self.img_size)
            annot[4] = bkp_class
        else:
            annot = np.multiply(annot, self.img_size)

        return annot

    def validate_bbox(self,annot,annot_type=None):
        if annot_type  == 'yolo':
            annot = self.from_yolo_to_default(annot)
        else:
            raise Exception('undefined annot_type')

        topleft_x,topleft_y,width,height = annot[0],annot[1],annot[2],annot[3]

        if (topleft_x < 0 or topleft_x + width > self.img_size) or (topleft_y < 0 or topleft_y + height > self.img_size) :
            print('topleft_x,topleft_y,width,height -> ', topleft_x,topleft_y,width,height)
            raise Exception('bbox does not fit to image dimensions')

    def convertDefaultAnnotToCoord(self, annot):
        raise Exception('Check normalizations')
        '''
        annot -> [x, y, w, h]
        '''

        w = annot[2] * self.WIDTH
        h = annot[3] * self.HEIGHT

        x = annot[0] * self.HEIGHT
        y = annot[1] * self.HEIGHT

        return [x,y,w,h]

    def convertYoloAnnotToCoord(self, yolo_annot):
        raise Exception('Check normalizations')
        '''
        yolo_annot -> [x, y, w, h]
        '''
        print(yolo_annot,self.WIDTH)
        w = yolo_annot[2] * self.WIDTH
        h = yolo_annot[3] * self.HEIGHT

        x = (yolo_annot[0] * self.WIDTH) - (w/2)
        y = (yolo_annot[1] * self.HEIGHT) - (h/2)

        return [x,y,w,h]

    def show_generated(self):
        fig = plt.figure(figsize=(12, 3))
        fig.subplots_adjust(top=0.85)
        fig.suptitle('Samples from the dataset.')
        legend_plotted = False
        for i_subplot in range(1, 6):
            i_img = i_subplot-1
            plt.subplot(1, 5, i_subplot)
            # plt.imshow(self.imgs[i_img],cmap=plt.cm.gray)
            plt.imshow(self.imgs[i_img],cmap='Greys', interpolation='none', origin='lower', extent=[0, self.img_size, 0, self.img_size])

            for i_obj, obj_y in enumerate(self.y[i_img]):
                x,y,w,h,gt_class = obj_y[0], obj_y[1], obj_y[2], obj_y[3], int(obj_y[4])

                gt_bbox = self.from_yolo_to_default([x,y,w,h])
                print('img {} obj {} bbox {} class {}'.format(i_img, i_obj, gt_bbox, gt_class))
                # plt.gca().set_ylim(0, self.img_size)
                # plt.gca().set_xlim(0, self.img_size)
                plt.gca().add_patch(matplotlib.patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], ec='b', fc='none'))
                plt.annotate(
                    '{}'.format(self.shape_labels[gt_class]),
                    (gt_bbox[0], gt_bbox[1]+gt_bbox[3]+0.2),
                    color=self.shape_labels_colors[gt_class])
                if not legend_plotted:
                    legend_plotted = True
                    plt.gca().legend(['GT'],loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
        plt.show()

    def plot_rectangle(self, img, bbox):

        fig = plt.figure()
        fig.suptitle('Plotting rectangle.')
        fig.subplots_adjust(top=0.85)

        plt.subplot(1, 1, 1)
        plt.imshow(img, cmap='Greys', interpolation='none', origin='lower', extent=[0, self.WIDTH, 0, self.HEIGHT])
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))
        plt.show()

    def show_predicted(self, predictions, gt, show_gt=False):
        fig = plt.figure(figsize=(12, 3))
        fig.subplots_adjust(top=0.85)
        fig.suptitle('Prediction demonstration. Random samples.')
        legend_plotted = False
        for i_subplot in range(1, 6):
            plt.subplot(1, 5, i_subplot)
            plt.imshow(self.test_imgs[i_subplot-1], cmap='Greys', interpolation='none', origin='lower', extent=[0, self.img_size, 0, self.img_size])

            if show_gt:
                for gt_data in gt[i_subplot-1]:
                    gt_bbox = self.from_yolo_to_default([gt_data[0], gt_data[1], gt_data[2], gt_data[3]])
                    plt.gca().add_patch(matplotlib.patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], ec='b', fc='none', lw=1.0, ls='solid'))

            for pred_data in predictions[i_subplot-1]:
                x,y,w,h,pred_class = pred_data[0], pred_data[1], pred_data[2], pred_data[3], pred_data[4]
                pred_bbox = self.from_yolo_to_default([x,y,w,h])

                plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none',lw=1.0,ls='solid'))

                plt.annotate('{}'.format(self.shape_labels[pred_class]), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2),
                    color=self.shape_labels_colors[pred_class])
                if not legend_plotted:
                    legend_plotted = True
                    plt.gca().legend(['Pred','GT'],loc='upper center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
        plt.show()

    def grv_mean_iou(self,pred,gt):
        print('Calculating IOU.')
        print('#WARNING: This function needs to be improved. There may be a way to achieve a better iou when relating gt x pred bboxes.')
        pred = np.copy(pred)
        gt = np.copy(gt)

        UNAVAILABLE_FLAG = 1
        pred_discard_control = []
        for p_bboxes in pred:
            '''
            Build a list of zeros according to the number of pred bboxes.
            Each image prediction can output a different number of bboxes. That is not good
            to work in numpy, mainly because the general shape is undefined.

            We put a zero for each predicted bbox, meaning they are available.
            Putting a UNAVAILABLE_FLAG in a determinated position means that bbox is unavailable.
            '''
            pred_discard_control.append(np.zeros((len(p_bboxes))))

        iou_scores = [] #average iou per image

        # get the gts for every image
        for img_index, img_gt_bboxes in enumerate(gt):
            img_iou_scores = []

            #get the gts for a specific image
            for gt_bbox in img_gt_bboxes:
                #holds iou scores for all predictions for this image in relation to this gt.
                gt_bbox_iou_scores = []

                #get the predicitions for the same image
                for pred_index, pred_bbox in enumerate(pred[img_index]):
                    #check availability
                    if pred_discard_control[img_index][pred_index] == UNAVAILABLE_FLAG:
                        continue

                    #calculate the iou of all predictions for this gt.
                    iou = self.bbox_iou_centered(pred_bbox, gt_bbox)
                    # print('comparing pred, gt, iou',pred_bbox,gt_bbox,iou)
                    gt_bbox_iou_scores.append(iou)

                # if there are usable predicitions.
                if len(gt_bbox_iou_scores) > 0:
                    # here we find the best predicition for this gt.
                    # print('gt_bbox_iou_scores',gt_bbox_iou_scores)
                    best_pred = np.argmax(gt_bbox_iou_scores)
                    # print('for gt_bbox the best_iou',gt_bbox, gt_bbox_iou_scores[best_pred])
                    img_iou_scores.append(gt_bbox_iou_scores[best_pred]) #save the best iou for the gt

                    #Mark as unavailable, so that it cannot be reused for other gts
                    pred_discard_control[img_index][best_pred] = UNAVAILABLE_FLAG

            #now we average the iou scores for this image and save it.
            iou_scores.append(np.average(img_iou_scores))

        return np.average(iou_scores),iou_scores

    def grv_mean_iou_old(self,pred,gt):
        print('Calculating IOU.')
        print('#This function needs to be improved. There may be a way to achieve a better iou when relating gt x pred bboxes.')
        _pred = np.copy(pred)
        gt = np.copy(gt)

        #add an extra column as flag. If the value is different than zero the bbox should not be considered anymore
        print('_pred.shape',_pred.shape)
        control_column_idx = _pred.shape[2]
        DISCARDED_FLAG = 1
        pred = np.zeros((_pred.shape[0],_pred.shape[1],control_column_idx+1))
        pred[:,:,:-1] = _pred

        iou_scores = [] #average iou per image

        # get the gts for every image
        for img_index, img_gt_bboxes in enumerate(gt):
            img_iou_scores = []

            #get the gts for a specific image
            for gt_bbox in img_gt_bboxes:
                #holds iou scores for all predictions for this image in relation to this gt.
                gt_bbox_iou_scores = []

                #get the predicitions for the same image
                for pred_bbox in pred[img_index]:
                    if pred_bbox[control_column_idx] == DISCARDED_FLAG:
                        continue

                    #calculate the iou of all predictions for this gt.
                    iou = self.bbox_iou_centered(pred_bbox, gt_bbox)
                    # print('comparing pred, gt, iou',pred_bbox,gt_bbox,iou)
                    gt_bbox_iou_scores.append(iou)

                # if there are usable predicitions.
                if len(gt_bbox_iou_scores) > 0:
                    # here we find the best predicition for this gt.
                    # print('gt_bbox_iou_scores',gt_bbox_iou_scores)
                    best_pred = np.argmax(gt_bbox_iou_scores)
                    # print('for gt_bbox the best_iou',gt_bbox, gt_bbox_iou_scores[best_pred])
                    img_iou_scores.append(gt_bbox_iou_scores[best_pred]) #save the best iou for the gt

                    #Mask to discard, so that it cannot be reused for other gts
                    pred[img_index][best_pred][control_column_idx] = DISCARDED_FLAG

            #now we average the iou scores for this image and save it.
            iou_scores.append(np.average(img_iou_scores))

        return np.average(iou_scores),iou_scores
