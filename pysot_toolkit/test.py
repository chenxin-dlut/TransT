# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
print(env_path)
if env_path not in sys.path:
    sys.path.append(env_path)
import argparse
import os

import cv2
import torch
import numpy as np

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
# from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone

parser = argparse.ArgumentParser(description='transt tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--name', default='', type=str,
        help='name of results')
args = parser.parse_args()

# torch.set_num_threads(1)


def main():
    # load config

    dataset_root = '/root/trans-t/data/test' #Absolute path of the dataset
    net_path = '/root/trans-t/data/model/transt.pth' #Absolute path of the model

    # create model
    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.name
    total_lost = 0

    for v_idx, video in enumerate(dataset):

        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tic = cv2.getTickCount()
            if idx == 0:

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))

                gt_bbox_ = [cx-w/2, cy-h/2, w, h]

                init_info = {'init_bbox': gt_bbox_}
                tracker.initialize(img, init_info)

                pred_bbox = gt_bbox_
                scores.append(None)

                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['target_bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])

            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())

            if idx == 0:
                cv2.destroyAllWindows()

            if args.vis:
                visualize_path = './visualize_tracks/'

                os.makedirs(visualize_path, exist_ok=True)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(visualize_path, f'{idx:06d}.png'), img)

                # cv2.waitKey(1)

        toc /= cv2.getTickFrequency()
        # save results
        if 'VOT2018-LT' == args.dataset:
            video_path = os.path.join('results', args.dataset, model_name,
                    'longterm', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path,
                    '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            result_path = os.path.join(video_path,
                    '{}_001_confidence.value'.format(video.name))
            with open(result_path, 'w') as f:
                for x in scores:
                    f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
            result_path = os.path.join(video_path,
                    '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))
        elif 'GOT-10k' == args.dataset:
            video_path = os.path.join('results', args.dataset, model_name, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            result_path = os.path.join(video_path,
                    '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))
        else:
            model_path = os.path.join('results', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
