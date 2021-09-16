import cv2

import sly_functions
import sly_globals as g

import supervisely_lib as sly


class TrackerContainer:
    def __init__(self, context, api):
        self.api = api
        
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]

        self.geometries = []
        self.frames_indexes = []

        self.add_geometries()
        self.add_frames_indexes()

        g.logger.info(f'TrackerController Initialized')

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += (1 if self.direction == 'forward' else -1)

    def track(self):
        images_cache = {}
        current_progress = 0
        notify_every = sly_functions.calculate_nofity_step(self.frames_count)

        all_figures = self.figure_ids.copy()
        all_objects = self.object_ids.copy()
        all_geometries = self.geometries.copy()

        for single_figure, single_object, single_geometry in zip(all_figures, all_objects, all_geometries):
            figure_ids = [single_figure]
            object_ids = [single_object]
            geometries = [single_geometry]

            states = [None for _ in range(len(figure_ids))]
            frame_start = None

            for enumerate_frame_index, frame_index in enumerate(self.frames_indexes):
                if frame_start is None:
                    frame_start = frame_index

                img_bgr = sly_functions.get_frame_np(self.api, images_cache, self.video_id, frame_index)
                im_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_height, img_width = img_bgr.shape[:2]


                if enumerate_frame_index == 0:
                    for i, (object_id, figure_id, geometry) in enumerate(zip(object_ids, figure_ids, geometries)):
                        bbox = sly_functions.convert_sly_geometry_to_lt_wh(geometry)
                        state = sly_functions.init_tracker(im_rgb, bbox)
                        states[i] = state
                        current_progress += 1

                else:
                    for i, (object_id, figure_id, geometry) in enumerate(zip(object_ids, figure_ids, geometries)):
                        state = states[i]

                        state, bbox_predicted = sly_functions.track(state, im_rgb)
                        states[i] = state

                        bbox_predicted = sly_functions.validate_figure(img_height, img_width, bbox_predicted)
                        created_figure_id = self.api.video.figure.create(self.video_id,
                                                                      object_id,
                                                                      frame_index,
                                                                      bbox_predicted.to_json(),
                                                                      bbox_predicted.geometry_name(),
                                                                      self.track_id)

                        current_progress += 1
                        if (current_progress % notify_every == 0 and enumerate_frame_index != 0) or frame_index == \
                                self.frames_indexes[-1]:
                            need_stop = self.api.video.notify_progress(self.track_id, self.video_id,
                                                                    min(frame_start, frame_index),
                                                                    max(frame_start, frame_index),
                                                                    current_progress,
                                                                    len(self.frames_indexes) * len(all_figures))
                            frame_start = None
                            if need_stop:
                                g.logger.debug('Tracking was stopped', extra={'track_id': self.track_id})
                                break
                g.logger.info(f'Process frame {enumerate_frame_index} — {frame_index}')
        g.logger.info(f'Tracking completed')

