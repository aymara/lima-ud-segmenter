#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from importlib import import_module


def get_model(name):
    mod = import_module('deeplima.segmenter.models.%s' % name)
    return mod.build_fn