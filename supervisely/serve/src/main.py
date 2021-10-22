import functools

import sly_globals as g
import supervisely_lib as sly

from tracker import TrackerContainer

import torch


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("ping")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    pass


@g.my_app.callback("track")
@sly.timeit
# @send_error_data
def track(api: sly.Api, task_id, context, state, app_logger):
    tracker = TrackerContainer(context, api)
    tracker.track()


def main():
    
    if torch.cuda.is_available():    
        sly.logger.info("🟩 Model has been successfully deployed")

        sly.logger.info("Script arguments", extra={
            "context.teamId": g.team_id,
            "context.workspaceId": g.workspace_id
        })
        g.my_app.run()
    else:
        sly.logger.info("🟥 GPU is not available, please run on agent with GPU and CUDA!")


if __name__ == "__main__":
    sly.main_wrapper("main", main)

    # track({  # for debug
    #     "teamId": 11,
    #     "workspaceId": 32,
    #     "videoId": 1114885,
    #     "objectIds": [236670],
    #     "figureIds": [54200821],
    #     "frameIndex": 0,
    #     "direction": 'forward',
    #     'frames': 10,
    #     'trackId': '5b82a928-0566-4d4d-a8e3-35f5abc736fe',
    #     'figuresIds': [54200821]
    # })
