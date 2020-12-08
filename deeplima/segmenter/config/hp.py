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

import math


def guess_hp(args, config, ds):
    input_width = 0

    for i in range(len(config['encoding']['ngrams'])):
        dict_size = ds.ngrams().size(i)
        embd_dim = int(math.ceil(math.sqrt(math.sqrt(dict_size))))
        if embd_dim % 2 == 1:
            embd_dim += 1
        config['encoding']['ngrams'][i]['embd_size'] = embd_dim + 4
        input_width += embd_dim

    hidden_dim = input_width if input_width % 2 == 0 else input_width + 1
    hidden_dim *= 2

    config['model']['rnn'][0]['fw_dim'] = hidden_dim
    config['model']['rnn'][0]['bw_dim'] = hidden_dim
    if len(config['model']['rnn']) > 1:
        hidden_dim = hidden_dim / 2 if hidden_dim % 2 == 0 else (hidden_dim + 1) / 2
        hidden_dim = hidden_dim if hidden_dim >= len(config['encoding']['tags']) else len(config['encoding']['tags'])
        hidden_dim = hidden_dim if hidden_dim % 2 == 0 else hidden_dim + 1
        config['model']['rnn'][1]['fw_dim'] = int(hidden_dim)
        config['model']['rnn'][1]['bw_dim'] = int(hidden_dim)
