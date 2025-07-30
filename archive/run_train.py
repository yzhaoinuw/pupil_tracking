# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 16:47:46 2025

@author: yzhao
"""

import deeplabcut


deeplabcut.create_new_project(
    "PupilTracking",
    "YueZhao",
    [
        "C:/Users/yzhao/python_projects/pupil_tracking/data/250616_5120_Purple_sleep_trial 1_2025-06-16T16-31-19.701_0000.png"
    ],
    copy_videos=False,
)
"""
#%%
config_path = 'C:/Users/yzhao/python_projects/pupil_tracking/PupilTracking-YueZhao-2025-07-23/config.yaml'
deeplabcut.check_labels(config_path)
#deeplabcut.convertcsv2h5(config_path)
deeplabcut.create_training_dataset(config_path)

#%%
deeplabcut.train_network(
    config_path,
    #shuffle=1,
    #trainingsetindex=0,
    device="cuda:0",
    max_snapshots_to_keep=5,
    displayiters=100,
    save_epochs=5,
    epochs=100,
)
"""
