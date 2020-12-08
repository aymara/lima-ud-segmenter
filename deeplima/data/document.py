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


class Document:
    def __init__(self, *args):
        self._text = ''       # plain text
        self._tokens = []     # tokens
        self._sentences = []  # bottom-level segments

        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], list):
            self._text = args[0]
            # ...
            pass

    def text(self):
        return self._text

    def sentences(self):
        return self._sentences

    def tokens(self):
        return self._tokens
