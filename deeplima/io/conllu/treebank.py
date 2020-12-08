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

import os
import re
import sys

from conllu import parse

from deeplima.data.document import Document
from deeplima.data.sentence import Sentence
from deeplima.data.token import Token
from deeplima.data.word import Word


class Treebank:
    def __init__(self, *args):
        self.sets = {}
        if 2 <= len(args) <= 3:
            base_path = args[0]
            treebank_name = args[1]
            set_ids = ('train', 'dev')
            if len(args) > 2:
                set_ids = args[2]
            self.load(base_path, treebank_name, set_ids)

    def set(self, name):
        return self.sets[name]

    def load(self, base_path, treebank, set_ids=('train', 'dev')):
        full_path = os.path.join(base_path, treebank)
        base_name = self._guess_base_files_name(full_path)
        if base_name is None:
            raise

        prefix = os.path.join(full_path, base_name)
        for n in set_ids:
            file_name = '%s-%s.conllu' % (prefix, n)
            try:
                file = open(file_name, "r", encoding="utf-8")
            except Exception as e:
                sys.stderr.write('Can\'t open file %s\n' % file_name)
                raise e

            parsed_data = parse(file.read())

            file_name = '%s-%s.txt' % (prefix, n)
            try:
                plain_text = open(file_name, "r", encoding="utf-8").read()
                # optional
                #plain_text = plain_text.replace('\n', ' ').replace('\r', ' ')
            except Exception as e:
                sys.stderr.write('Can\'t open file %s\n' % file_name)
                raise e

            self.sets[n] = [self._create_document(plain_text, parsed_data)]

    @staticmethod
    def _guess_base_files_name(path):
        for (dirpath, dirnames, filenames) in os.walk(path):
            for fn in filenames:
                if fn.endswith('.conllu'):
                    base = os.path.basename(fn)
                    mo = re.match('^(.*?)-test.conllu', base)
                    if mo:
                        return mo.group(1)
        return None

    @staticmethod
    def _is_space(text, idx):
        ch = text[idx:idx + 1]
        return ch.isspace()

    def _align(self, text, idx, token):
        while self._is_space(text, idx) and idx < len(text):
            idx += 1

        start = idx
        stop = idx + len(token['form'])
        form_in_source = text[idx:stop].replace('\n', ' ')
        if form_in_source == token['form']:
            return start, stop

        return None, -1

    def _create_document(self, plain_text, parsed_data):
        doc = Document()
        doc._text = plain_text

        idx = 0
        mwt = None
        for sent in parsed_data:
            first_token_id = None
            tokens_in_sentence = 0
            for token in sent:
                id_str = str(token['id'])
                if '.' in id_str:
                    continue  # these words don't exist in source text

                if mwt is None:
                    start, idx = self._align(plain_text, idx, token)
                    if idx == -1:
                        raise

                    doc.tokens().append(Token(doc, start, len(token['form']), None if mwt is None else []))

                    if first_token_id is None:
                        first_token_id = len(doc.tokens()) - 1
                    tokens_in_sentence += 1
                else:
                    if not id_str.isdecimal():
                        raise

                    id_int = int(token['id'])

                    if mwt['first_id'] <= id_int <= mwt['last_id']:
                        doc.tokens()[-1].words().append(Word(doc.tokens()[-1], token['form']))

                    if id_int == mwt['last_id']:
                        mwt = None

                if '-' in id_str:
                    mwt = {
                        'first_id': int(token['id'][0]),
                        'last_id': int(token['id'][2])
                    }

            doc.sentences().append(Sentence(doc, first_token_id, tokens_in_sentence))

        return doc
