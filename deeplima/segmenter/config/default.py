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

config = {
    'encoding': {
        'mode': 'tokenize&split',
        'padding_len': 4,
        'ngrams': [
            {
                'start': 0,
                'len': 1,
                'min_freq': 0.5
            },
            {
                'start': -1,
                'len': 2,
                'min_freq': 1
            },
            {
                'start': -1,
                'len': 3,
                'min_freq': 2
            }
        ],
        'tags': [
            'B', # token: begin
            'I', # token: inside
            'E', # token: end
            'S', # token: single-character token
            'X', # token: outside of token
            'T', # sentence: last single-character token in sentence
            'U'  # sentence: last multi-character token in sentence
         ],
        'UNK': '<UNK>'
    },
    'dicts': {
        'chars': None
    },
    'hp': {
        'max_seq_len': 400,
        'batch_size': 100,
        'embd_l2': 0.1,
        'learning_rate': 0.01,
        'learning_rate_decay': 0.1,
        'input_keep_prob': 0.5,
        'rnn_output_keep_prob': 0.5,
        'dense_keep_prob': 1.0,
    },
    'model': {
        'name': 'birnncrf',
        'rnn': [
            {
                'type': 'lstm',
                'fw_dim': 10,
                'bw_dim': 10
            },
            {
                'type': 'lstm',
                'fw_dim': 10,
                'bw_dim': 10
            }
        ],
        'dense': [
            {
                'dim': None
            }
        ]
    }
}