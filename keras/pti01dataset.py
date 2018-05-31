import glob, os, random
import numpy as np
from keras.preprocessing.image import load_img, array_to_img, img_to_array

class PTI01Dataset:
    def __init__(self, dataset_path = None, batch_size = 32, epochs = 2, train_split = 0.5, img_rescale=None, shuffle=True):

        if not dataset_path:
            raise Exception('no dataset path defined.')

        if img_rescale and not isinstance(img_rescale, float):
            raise Exception('img_rescale must be float.')

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_split = train_split
        self.img_rescale = img_rescale
        self.shuffle = shuffle

        self.img_width_original = 640
        self.img_height_original = 480
        self.img_width = 0
        self.img_height = 0
        self.img_channels = 3

        if self.img_rescale:
            self.img_width = int(self.img_width_original * self.img_rescale)
            self.img_height = int(self.img_height_original * self.img_rescale)
            print('The images are going to be rescaled to {}x{}'.format(self.img_width,self.img_height))
        else:
            self.img_width = self.img_width_original
            self.img_height = self.img_height_original

        self.read_dataset()

    def get_img_shape(self,original=False):
        if original:
            return (self.img_height, self.img_width, self.img_channels)
        return (self.img_height_original, self.img_width_original, self.img_channels)

    def read_dataset(self):
        print('Searching for images in the dataset path...')
        image_list_check = glob.glob(os.path.join(self.dataset_path, '**/*.jpg'), recursive=True)
        image_list_size = len(image_list_check)
        if image_list_size <= 0:
            print('...No images have been found in the given dataset path: {}'.format(self.dataset_path))
            raise Exception('No images.')
        print('Done. # of images found: {}'.format(image_list_size))

        if self.shuffle:
            print('Shuffling image list...')
            random.shuffle(image_list_check)
            print('Done.')

        print('Locating the annotation files...')
        self.image_path_list = []
        Y = []
        no_label = 0
        for img in image_list_check:
            label_path = img.replace('.jpg','.txt')

            if not os.path.exists(label_path):
                no_label += 1
            else:
                self.image_path_list.append(img)

                with open(label_path) as f:
                    bboxes = []
                    for line in f:
                        #expecting yolo annotation format.
                        data = [float(t.strip()) for t in line.split()]
                        if data[0] == 0.0: #pedestrian
                            data.pop(0) # remove class
                            bboxes.append(data)
                    Y.append(bboxes)
        self.Y = np.array(Y)
        del image_list_check

        self.image_path_list_size = len(self.image_path_list)
        if no_label > 0:
            print('...There are {} images with no annotations...'.format(no_label))
        print('Done. Final number of images/annotation files found: {}/{}'.format(self.image_path_list_size,len(Y)))


        self.train_size = int(self.train_split * self.image_path_list_size)
        self.test_size = self.image_path_list_size - self.train_size
        assert (self.train_size + self.test_size) == self.image_path_list_size

        self.steps_per_epoch = self.train_size / self.epochs

    def get_steps_per_epoch(self):
        return self.train_size / self.epochs

    def create_train_gen(self):
        curr_idx = 0
        train_X, train_Y = self.get_train_slice()
        train_X_len = len(train_X)

        while 1:
            if curr_idx >= train_X_len:
                print('Dataset train gen: reseting.')
                curr_idx = 0

            to_idx = curr_idx + self.batch_size
            if to_idx >= train_X_len:
                print('Not enough samples for batch. Using {}'.format(to_idx - curr_idx))
                to_idx = train_X_len

            Y_batch = train_Y[curr_idx:to_idx] # ready

            X_paths_batch = train_X[curr_idx:to_idx]
            X_batch = []
            for im_path in X_paths_batch:
                im = load_img(im_path, grayscale=False)
                if self.img_rescale:
                    im = cv2.resize(img_to_array(im), (self.img_width,self.img_height))
                    X_batch.append(im)
                else:
                    X_batch.append(img_to_array(im)) #not sure about this


            yield (np.array(X_batch), Y_batch)

    def get_train_slice(self):

        train_X_paths = self.image_path_list[:self.train_size]
        # train_Y_paths = self.image_path_list[self.train_size:]

        pool_train_Y = np.array(self.Y[:self.train_size])
        train_Y = np.zeros((len(pool_train_Y), 1, 4))

        for i in range(self.train_size):
            y = pool_train_Y[i]
            if len(y) >= 1:
                train_Y[i][0] = y[0]

        train_Y = train_Y.reshape(len(pool_train_Y), -1)

        return train_X_paths, train_Y
