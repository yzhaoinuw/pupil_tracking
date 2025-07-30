# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 21:24:13 2025

@author: yzhao
"""

import deeplabcut

config_path = "C:/Users/yzhao/python_projects/pupil_tracking/PupilTracking-YueZhao-2025-07-23/config.yaml"
deeplabcut.evaluate_network(config_path, plotting=True)
