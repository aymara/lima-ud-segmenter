#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


class Sentence:
    def __init__(self, doc, start, length):
        self._doc = doc
        self._start = start
        self._len = length

    def start(self):
        return self._start

    def __len__(self):
        return self._len

    def tokens(self):
        for i in range(self._start, self._start + self._len):
            yield self._doc.tokens()[i]
