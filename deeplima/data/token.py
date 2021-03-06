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
