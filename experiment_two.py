
# coding: utf-8

# In[1]:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob, os, sys, random
import numpy as np
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator

# get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[2]:


#static
# DATASET_PATH = '/home/gustavo/workspace/datasets/pti/PTI01/'
DATASET_PATH = '/home/grvaliati/workspace/datasets/pti/PTI01/'
IMG_WIDTH_ORIGINAL = 640
IMG_HEIGHT_ORIGINAL = 480
IMG_WIDTH = 0
IMG_HEIGHT = 0

IMG_RESCALE = True
IMG_RESCALE_BY = 0.2
if IMG_RESCALE:
    IMG_WIDTH = int(IMG_WIDTH_ORIGINAL * IMG_RESCALE_BY)
    IMG_HEIGHT = int(IMG_HEIGHT_ORIGINAL * IMG_RESCALE_BY)
    print('The images are going to be rescaled to {}x{}'.format(IMG_WIDTH,IMG_HEIGHT))
else:
    IMG_WIDTH = IMG_WIDTH_ORIGINAL
    IMG_HEIGHT = IMG_HEIGHT_ORIGINAL


# SHAPE = (IMG_HEIGHT,IMG_WIDTH,3)
BATCH_SIZE = 2
EPOCHS = 30
TRAIN_SPLIT = 0.5 # amount of the dataset used for training.
SHUFFLE = True


# In[3]:


image_list_check = glob.glob(os.path.join(DATASET_PATH, '**/*.jpg'), recursive=True)
if SHUFFLE:
    random.shuffle(image_list_check)


# In[4]:


image_path_list = []
gt_list = []
for img in image_list_check:
    label_path = img.replace('.jpg','.txt')

    if not os.path.exists(label_path):
        print('Image has no label file: {}'.format(img))
    else:
        image_path_list.append(img)

        with open(label_path) as f:
            bboxes = []
            for line in f:
                #expecting yolo annotation format.
                data = [float(t.strip()) for t in line.split()]
                if data[0] == 0.0: #pedestrian
                    data.pop(0) # remove class
                    bboxes.append(data)
            gt_list.append(bboxes)

Y = np.array(gt_list)
# Y = np.array(gt_list).reshape(len(gt_list), -1)
del image_list_check
dataset_size = len(image_path_list)
print('Loaded {} img paths and {} gts'.format(dataset_size,len(gt_list)))


import cv2

X = []

print("Loading images in memory.")
for i,img_path in enumerate(image_path_list):
    # if i > 1000:
        # break
    im = load_img(img_path, grayscale=False)

    # X.append(img_to_array(im).reshape(-1))
    # X.append(img_to_array(im))

    if IMG_RESCALE:
        im = cv2.resize(img_to_array(im), (IMG_WIDTH,IMG_HEIGHT))
        X.append(im)
    else:
        X.append(img_to_array(im)) #not sure about this

#     X.append(im.reshape(-1))
print("Parsing to np.array")
X = np.array(X) #GASTANDO MEMÓRIA

print("done.")

t_slice = int(TRAIN_SPLIT * len(X))

train_X = np.array(X[:t_slice])
train_X_paths = image_path_list[:t_slice]

pool_train_Y = np.array(Y[:t_slice])
train_Y = np.zeros((len(pool_train_Y), 1, 4))
train_Y.shape

print('Getting single bbox for gt training')
for i in range(t_slice):
    y = pool_train_Y[i]
    if len(y) >= 1:
        train_Y[i][0] = y[0]
    else:
        print("image with no bboxes")

train_Y = train_Y.reshape(len(pool_train_Y), -1)
print('Done.')

test_X = np.array(X[t_slice:])
test_X_paths = image_path_list[t_slice:]
# test_Y = np.array(Y[t_slice:len(X)])

pool_test_Y = np.array(Y[t_slice:len(X)])
test_Y = np.zeros((len(pool_test_Y), 1, 4))
for i in range(t_slice):
    y = pool_test_Y[i]
    if len(y) >= 1:
        test_Y[i][0] = y[0]
    else:
        print("image with no bboxes")

test_Y = test_Y.reshape(len(pool_test_Y), -1)

print('Train/Test: {}/{}'.format(len(train_X),len(test_X)))

test_img = image_path_list[t_slice:]
test_bboxes = gt_list[t_slice:]


# In[6]:


def convertYoloAnnotToCoord(yolo_annot):

    w = yolo_annot[2] * IMG_WIDTH_ORIGINAL
    h = yolo_annot[3] * IMG_HEIGHT_ORIGINAL

    x = (yolo_annot[0] * IMG_WIDTH_ORIGINAL) - (w/2)
    y = (yolo_annot[1] * IMG_HEIGHT_ORIGINAL) - (h/2)

    return [x,y,w,h]


# In[7]:


rand_img_idx = random.randint(0,len(X)-1)

im = load_img(image_path_list[rand_img_idx], grayscale=False)
plt.imshow(array_to_img(im))
for bbox in gt_list[rand_img_idx]:
    bbox = convertYoloAnnotToCoord(bbox)
    print(bbox)
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))


# In[10]:


# Build the model.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Input
from keras.optimizers import SGD
# model = Sequential([
#         Dense(200, input_dim=train_X.shape[-1]),
#         Activation('relu'),
#         Dropout(0.2),
#         Dense(4)
#     ])

model = Sequential([
    Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(name='norm_1'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1000, activation='relu'),
    Dense(4)
])
model.compile('adadelta', 'mse')

model.summary()

from keras.models import load_model

model_name = 'model.h5'

if os.path.exists('model.h5'):
    print('Loading pre-existing model from disk.')
    model = load_model(model_name)
else:
    print('Training...')
    model.fit(train_X, train_Y, validation_split=0.2, epochs=EPOCHS, verbose=2)
    model.save(model_name)

print('predicting...')
pred_y = model.predict(test_X[:10])
print('done.')
# pred_bboxes = pred_Y[:1] * img_size
print(pred_y, pred_y.shape)
# pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
# pred_bboxes.shape
os.makedirs('./predictions',exist_ok=True)
for t_index in range(10):#len(test_X)
    print('reading predictions')
    im = load_img(image_path_list[t_index], grayscale=False)
    plt.close()
    plt.imshow(array_to_img(im))
    for bbox in gt_list[t_index]:
        bbox = convertYoloAnnotToCoord(bbox)
        # print(bbox)
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='g', fc='none'))

    bbox = convertYoloAnnotToCoord(pred_y[t_index])
    print('pred bbox img {}:{}'.format(t_index, bbox))
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))

    plt.savefig('predictions/prediction_img{}.jpg'.format(t_index), bbox_inches='tight')
    plt.show()
    plt.close()
