#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

config = {
    'encoding': {
        'mode': 'tokenize&split',
        'padding_len': 4,
        'ngrams': [
            {
                'start': 0,
                'len': 1,
                'min_freq': 6
            },
            {
                'start': -1,
                'len': 2,
                'min_freq': 10
            },
            {
                'start': -1,
                'len': 3,
                'min_freq': 30
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