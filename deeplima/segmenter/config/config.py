#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Copyright 2018-2020 CEA LIST
#
# This file is part of LIMA.
#
# LIMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LIMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with LIMA.  If not, see <https://www.gnu.org/licenses/>

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
