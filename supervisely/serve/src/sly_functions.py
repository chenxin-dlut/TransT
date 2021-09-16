from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import supervisely_lib as sly
import sly_globals as g

import numpy as np

from pysot_toolkit.bbox import get_axis_aligned_bbox
from pysot_toolkit.toolkit.datasets import DatasetFactory
# from pysot_toolkit.toolkit.utils.region import vot_overlap, vot_float2str
from pysot_toolkit.trackers.tracker import Tracker
from pysot_toolkit.trackers.net_wrappers import NetWithBackbone


def _conver_bbox_from_sly_to_opencv(bbox):
    opencv_bbox = [bbox.left, bbox.top, bbox.width, bbox.height]
    return opencv_bbox


def init_tracker(img, bbox):
    net_path = '/root/trans-t/data/model/transt.pth'  # Absolute path of the model

    # create model
    net = NetWithBackbone(net_path=net_path, use_gpu=True)
    tracker = Tracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

    cx, cy, w, h = get_axis_aligned_bbox(np.array(bbox))

    gt_bbox_ = [cx - w / 2, cy - h / 2, w, h]

    init_info = {'init_bbox': gt_bbox_}
    tracker.initialize(img, init_info)

    return tracker


def track(tracker, img):
    outputs = tracker.track(img)
    prediction_bbox = outputs['target_bbox']

    left = prediction_bbox[0]
    top = prediction_bbox[1]
    right = prediction_bbox[0] + prediction_bbox[2]
    bottom = prediction_bbox[1] + prediction_bbox[3]
    return tracker, sly.Rectangle(top, left, bottom, right)


def get_frame_np(api, images_cache, video_id, frame_index):
    uniq_key = "{}_{}".format(video_id, frame_index)
    if uniq_key not in images_cache:
        img_rgb = api.video.frame.download_np(video_id, frame_index)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        images_cache[uniq_key] = img_bgr
    return images_cache[uniq_key]


def validate_figure(img_height, img_width, figure):
    img_size = (img_height, img_width)
    # check figure is within image bounds
    canvas_rect = sly.Rectangle.from_size(img_size)
    if canvas_rect.contains(figure.to_bbox()) is False:
        # crop figure
        figures_after_crop = [cropped_figure for cropped_figure in figure.crop(canvas_rect)]
        if len(figures_after_crop) != 1:
            g.logger.warn("len(figures_after_crop) != 1")
        return figures_after_crop[0]
    else:
        return figure


def calculate_nofity_step(frames_count):
    notify_every_percent = 0.03
    return int(round(frames_count * notify_every_percent)) if int(round(frames_count * notify_every_percent)) > 0 else 1


def convert_sly_geometry_to_lt_wh(sly_geometry):
    bbox = sly_geometry.to_bbox()

    return [bbox.left, bbox.top, bbox.right - bbox.left, bbox.bottom - bbox.top]
