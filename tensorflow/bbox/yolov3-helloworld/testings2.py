from dataset import HelloWorldDataset
from model import translate_from_model_pred, translate_to_model_gt, Conv_net_01, Conv_net_02, Conv_net_03

IMG_SIZE = 16
CLASSES_NUMBER = 3

curr_model = Conv_net_02(img_size=IMG_SIZE, n_classes=CLASSES_NUMBER)

dataset = HelloWorldDataset(
    num_objects=2,
    shape_number=CLASSES_NUMBER,
    img_size=IMG_SIZE,
    train_proportion=0.8,
    num_imgs=30)

train_data, train_y, test_data, test_y = dataset.generate()
# dataset.show_generated()

for t_y_index, t_y in enumerate(test_y):
    print(t_y_index, t_y)

new_test_y = translate_to_model_gt(test_y,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True, verbose=True)

'''
Simulate our raw predicitions (from the network) are the new_test_y.
In this case our predictions would be perfect.
If our translation process is correct, we should plot perfect predictions.
'''
pred = translate_from_model_pred(new_test_y, curr_model.get_config(),verbose=True,obj_threshold=0.01)

for p in pred:
    print('p',p)

dataset.show_predicted(predictions=pred,gt=test_y)
