from dataset import HelloWorldDataset
from model import *

IMG_SIZE = 16
CLASSES_NUMBER = 3

curr_model = Conv_net_02(img_size=IMG_SIZE, n_classes=CLASSES_NUMBER)
# curr_model = Conv_net_01(img_size=IMG_SIZE, n_classes=CLASSES_NUMBER)

dataset = HelloWorldDataset(
    num_objects=2,
    shape_number=CLASSES_NUMBER,
    img_size=IMG_SIZE,
    train_proportion=0.8,
    num_imgs=10000,
    allow_overlap=True)

#The test_y is also going to be used as simulated perfect predictions.
# train_data, train_y, test_data, test_y = dataset.generate()
train_data, train_y, test_data, test_y = dataset.load_or_generate(dataset_path='./datasets')
# dataset.show_generated()

print('Printing the GT/PRED (limit max to 10)')
for t_y_index, t_y in enumerate(test_y):
    if t_y_index >= 10:
        break
    for t_obj_index, t_obj in enumerate(t_y):
        print("IMG {} OBJ {} ".format(t_y_index,t_obj_index), t_obj)

new_test_y = translate_to_model_gt(test_y,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True, verbose=False)

'''
Simulate our raw predicitions (from the network) are the new_test_y.
In this case our predictions would be perfect.
If our translation process is correct, we should plot perfect predictions.
'''
pred_translated = translate_from_model_pred(new_test_y, curr_model.get_config(),verbose=False,obj_threshold=0.01)
pred = do_nms(pred_translated, model_config=curr_model.get_config(), iou_func=dataset.bbox_iou_centered)

print('Camera ready predictions (first 10 imgs)...')
for img_index, img_p in enumerate(pred):
    for obj_index, obj_p in enumerate(img_p):
        if img_index >= 10:
            break
        for obj_gt in test_y[img_index]:
            iou = dataset.bbox_iou_centered(obj_gt,obj_p)
            print("For img {} - gt {} <-> pred {} has iou of {}".format(img_index,obj_gt,obj_p,iou))

mean_iou, iou_per_image = dataset.grv_mean_iou(pred,gt=test_y)
print('mean_iou',mean_iou)

dataset.show_predicted(predictions=pred,gt=test_y)
