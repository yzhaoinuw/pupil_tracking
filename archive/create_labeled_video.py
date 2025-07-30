# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 21:49:41 2025

@author: yzhao
"""

import deeplabcut

config_path = "C:/Users/yzhao/python_projects/pupil_tracking/PupilTracking-YueZhao-2025-07-23/config.yaml"
video_path = "C:/Users/yzhao/python_projects/pupil_tracking/movies/250616_5120_Purple_sleep_trial 1_2025-06-16T16-31-19.701.avi"

deeplabcut.create_labeled_video(
    config_path, [video_path], draw_skeleton=True, pcutoff=0.1
)
