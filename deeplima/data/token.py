#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from deeplima.data.word import Word


class Token:
    def __init__(self, doc, start, length, words=None):
        self._doc = doc
        self._start = start
        self._len = length
        if words is None:
            self._words = [Word(self)]
        else:
            self._words = words

    def text(self):
        stop = self._start + self._len + 1
        return self._doc.text()[self._start: stop]

    def words(self):
        return self._words

    def start(self):
        return self._start

    def __len__(self):
        return self._len
