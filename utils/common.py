# -*- coding: utf-8 -*-
# @Date    : 2020/9/20 16:58
# @Author  : Du Jing
# @FileName: common
# ---- Description ----


import time
import os
import sys


__all__ = [
    'ConfigDict',
    'Logger',
    'check_dir'
]


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class Logger:

    def __init__(self, stream=sys.stdout, log_name=None):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = log_name if log_name is not None else '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def check_dir(path):
    if path.__contains__('.'):
        res = input("Current path: \"%s\"\nwhether to continue? [y/n]: " % path)
        if res == 'n':
            return
    if not os.path.exists(path):
        parent = os.path.split(path)[0]
        check_dir(parent)
        os.mkdir(path)


