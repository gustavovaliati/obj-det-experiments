import matplotlib
# matplotlib.use('Agg')

import numpy as np
# import motmetrics as mm
# import pandas as pd
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
import datetime, os

from chainercv.datasets import voc_bbox_label_names
from chainercv.evaluations import eval_detection_voc, calc_detection_voc_prec_rec
from chainercv.datasets.voc import voc_utils

metric_plot_choices = [
    'all',
    'precision_recall_curve'
    ]

class Metrics:
    def __init__(self, data, metric = 'all', database_name = 'unnamed_database', plottings = 'all', model = 'unnamed_model',limit=None):
        self.metric = metric
        self.data = data
        self.database_name = database_name
        self.plottings = plottings
        self.model = model

    def plot(self, recall, precision, metric=None):

        if not metric:
            raise Exception('Metric name must be defined for plotting.')
        if not self.plottings:
            raise Exception('No plottings are specified.')

        if 'all' in self.plottings:
            print('Plotting everything.')
        else:
            for pl in self.plottings:
                if not pl in metric_plot_choices:
                    parser.error('The plotting you requested does not exist: {}.'.format(pl))
                    return

        fig_dir = "plottings/metrics/{}_{}_{}".format(self.model, self.database_name,
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(fig_dir, exist_ok=True)

        if 'all' in self.plottings:
            self.plottings = metric_plot_choices
            self.plottings.remove('all')

        if 'all' in self.plottings or 'precision_recall_curve' in self.plottings:
            self.plottings.remove('precision_recall_curve')
            print("Plotting {}.".format('precision_recall_curve'))

            print("Min prec {}, Max prec {}, Min rec {}, Max rec {}".format(min(precision),max(precision),min(recall),max(recall)))
            print("Avg prec {}, Avg rec {}".format(np.average(precision), np.average(recall)))
            plt.step(recall, precision, color='b', alpha=0.2,
                     where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2,
                             color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            #           average_precision))

            plt.savefig(os.path.join(fig_dir, metric+'__precision_recall_curve.jpg'), bbox_inches='tight')
            plt.show()
            plt.close()


        if len(self.plottings) > 0:
            raise Exception("There are unknown requested plottings: {}".format(self.plottings))

    def calc(self):
        pred_bboxes,pred_labels,pred_scores,gt_bboxes,gt_labels = self.data

        if self.metric == 'all' or self.metric == 'voc_detection':
            print('Calculating voc_detection ...')
            result = eval_detection_voc(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels,
                use_07_metric=True)
            print('AP: {:f}'.format(result['ap'][0]))
            # print('person mAP: {:f}'.format(result['ap'][voc_utils.voc_bbox_label_names.index('person')]))



        if self.metric == 'all' or self.metric == 'pr_voc_detection':
            print('Calculating pr_voc_detection ...')
            prec, rec = calc_detection_voc_prec_rec(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults=None, iou_thresh=0.5)


            prec = prec[0]
            rec = rec[0]

            # person_prec = prec[voc_utils.voc_bbox_label_names.index('person')]
            # person_rec = rec[voc_utils.voc_bbox_label_names.index('person')]

            print('Avg person precision: {:f}'.format(np.average(prec)))
            print('Avg person recall: {:f}'.format(np.average(rec)))

            # for i in range(25):
                # print(i, person_prec[i], person_rec[i])

            if self.plottings:
                self.plot(recall=rec, precision=prec, metric='pr_voc_detection')

            return prec,rec
