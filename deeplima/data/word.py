#!/usr/bin/env python3
# -*- coding: UTF-8 -*-


class Word:
    def __init__(self, token, text=None):
        self._token = token
        self._text = text

    def text(self):
        if self._text is None:
            return self._token.text()
        else:
            return self._text
