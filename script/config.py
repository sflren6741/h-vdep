# -*- coding: utf-8 -*-
from typing import List

import pandas as pd  # type: ignore

FIELD_LENGTH: float = 40  # unit: meters
FIELD_WIDTH: float = 20  # unit: meters
# DATE = "20250130_early_stopping_false_100iter_eval_metric_F1"
DATE = "20250128"
PREDICT_EVENT_NUM = 10
COLORS2 = ['#FF4B00', '#005AFF', '#03AF7A', '#FFFF00', '#00FFFF']
COLORS = ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#D55E00', '#56B4E9', '#CC79A7', '#000000']
H5_PATH = "./data/eigd/handball/positions/"
LABEL_COLS = [
    "concede_future_label", "foul_future_label", "fast_break_future_label",
]
CAT_COLS = ['event_type', 'result', 'possession_team', 'pre_event_type', 'pre_result']
UNUSED_COLS = ["possession_team", "result", "pre_result"]
EVENT_TYPE_LIST = [
    "pass",
    "reception",
    "throw_in",
    "sanction",
    "after_foul",
    "foul",
    "foul_penalty",
    "foul_free_throw",
    "shot",
    "throw_off",
    "goal",
    "ball_out",
    "possession_change",
    "other"
]

# result のユニークな値のリスト
RESULT_LIST = [
    "successful",
    "off target",
    "intercepted",
    "two_min",
    "yellow",
    "free_throw",
    "penalty",
    ""
]
