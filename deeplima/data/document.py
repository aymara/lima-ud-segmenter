#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


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
