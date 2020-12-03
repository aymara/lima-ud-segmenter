#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


class NgramDict:
    def __init__(self, args, chars):
        self._ngrams = []
        for n in args:
            self._ngrams.append(self._build_dict(n, chars))

    def find(self, idx, s):
        if s in self._ngrams[idx]['s2i']:
            return self._ngrams[idx]['s2i'][s]
        else:
            return self._ngrams[idx]['s2i']['<UNK>']

    def size(self, idx):
        return len(self._ngrams[idx]['i2s'])

    def to_save(self):
        return [ { 'i2w': x['i2s'] } for x in self._ngrams ]

    @staticmethod
    def _build_dict(args, chars):
        s2f = {}
        l = args['len']
        start = abs(args['start'])
        stop = len(chars) - args['len'] - args['start']
        for i in range(start, stop):
            s = chars[i:i+l]
            if s not in s2f:
                s2f[s] = 1
            else:
                s2f[s] += 1

        d = {
            's2i': { '<UNK>': 0 },
            'i2s': [ '<UNK>' ]
        }

        for s in sorted(s2f.keys()):
            ipm = (s2f[s] * 1000000) / len(chars)
            if ipm >= args['min_freq']:
                d['i2s'].append(s)
                d['s2i'][s] = len(d['i2s']) - 1

        return d