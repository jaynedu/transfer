# -*- coding: utf-8 -*-
# @Date    : 2020/9/20 16:59
# @Author  : Du Jing
# @FileName: data
# ---- Description ----


import os
import pandas as pd

__all__ = [
    'get_feature_name',
]


def get_feature_name(dir):
    filenames = os.listdir(dir)
    df = pd.read_csv(os.path.join(dir, filenames[0]))
    return df.columns.tolist()