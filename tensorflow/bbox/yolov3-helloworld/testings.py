from dataset import HelloWorldDataset

dummy_gt =    [
            [[3,3,2,3,0]], #bboxes for img1
            [[5,5,6,2,1]],  #bboxes for img2
            [[6,6,2,2,0]]
        ]
dummy_pred =  [
            [[4,3,2,2,0]],
            [[3,1,2,2,1]],
            []
        ]

#IOU
dataset = HelloWorldDataset(img_size=16)
#default image size 8x8
iou = dataset.bbox_iou_centered(
    dummy_gt[0][0],
    dummy_pred[0][0]
    )
assert iou == 0.25

iou = dataset.bbox_iou_centered(
    dummy_gt[1][0],
    dummy_pred[1][0]
    )
assert iou == 0.0


#MODEL
from model import translate_from_model_pred, translate_to_model_gt, Conv_net_02
curr_model = Conv_net_02(img_size=16)

#TEST ON DUMMY
new_dummy_gt = translate_to_model_gt(dummy_gt,curr_model.get_config(), dataset.bbox_iou_centered)

#lets say our new_dummy_gt is also our prediction, just to check our function.
translated_pred = translate_from_model_pred(new_dummy_gt , curr_model.get_config(),verbose=True)
print(translated_pred)
dataset.show_predicted(translated_pred, dummy_gt)

##TEST ON DATASET
# train_data, train_labels, eval_data, eval_labels = dataset.generate()
# print(train_labels[0])
#
# curr_model = Conv_net_02(img_size=dataset.img_size)
# new_train_labels = translate_to_model_gt(train_labels,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True)
# new_eval_labels = translate_to_model_gt(eval_labels,curr_model.get_config(), dataset.bbox_iou_centered, normalized=True)
