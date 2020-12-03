#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from importlib import import_module


def _build_indices(l):
    i2w = list(l)
    i2w.sort()
    w2i = {}
    for i in range(len(i2w)):
        w2i[i2w[i]] = i

    return i2w, w2i

def get_config(name='default'):
    mod = import_module('deeplima.segmenter.config.%s' % name)

    if 'model' in mod.config and 'dense' in mod.config['model']:
        last_dense_idx = len(mod.config['model']['dense']) - 1
        if mod.config['model']['dense'][last_dense_idx]['dim'] is None:
            mod.config['model']['dense'][last_dense_idx]['dim'] = len(mod.config['encoding']['tags'])

    if 'encoding' in mod.config and 'tags' in mod.config['encoding']:
        mod.config['encoding']['i2t'], mod.config['encoding']['t2i'] = _build_indices(mod.config['encoding']['tags'])

    return mod.config
