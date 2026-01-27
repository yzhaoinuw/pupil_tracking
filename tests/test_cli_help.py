# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:14:01 2026

@author: yzhao
"""

import subprocess


def test_cli_help_runs():
    # Uses the console script entrypoint you advertise in README
    # (works on all OS; if it fails on Windows CI later, we can adjust)
    proc = subprocess.run(
        ["run-pupil-analysis", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
