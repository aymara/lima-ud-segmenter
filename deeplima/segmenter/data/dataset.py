#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from random import sample, randint

from deeplima.segmenter.data.ngram import NgramDict

class Dataset:
    def __init__(self, args, docs, ngrams=None):
        padding_chars = ' ' * args['padding_len']
        padding_tags = 'X' * args['padding_len']
        self._chars = padding_chars
        self._tags = padding_tags

        for d in sample(docs, len(docs)):
            if len(self._chars) > args['padding_len']:
                self._chars += ' '
                self._tags += 'X'
            self._chars += d.text()
            self._tags += self._generate_tag(args, d)

        self._chars += padding_chars
        self._tags += padding_tags

        if ngrams is None:
            self._ngrams = NgramDict(args['ngrams'], self._chars)
        else:
            self._ngrams = ngrams

        self._input = self._annotate_chars(args, self._ngrams, self._chars)
        self._gold = np.array([ args['t2i'][x] for x in self._tags ], dtype=np.int32)
        self._cache = { 'batch': {} }

    def chars(self):
        return self._chars

    def ngrams(self):
        return self._ngrams

    def training_batch(self, length, batch_size):
        return self._batch(length, batch_size, 0, True, True)

    def prediction_batch(self, length, batch_size, start):
        for i in range(start, len(self._input[0]), length * batch_size):
            yield self._batch(length, batch_size, i, False, True)

    def _batch(self, length, batch_size, start_pos, random_batch, add_gold):
        total_chars = len(self._input[0])

        cache_id = '%d x %d' % (batch_size, length)
        if add_gold:
            cache_id += ' + gold'

        if 'batch' in self._cache and cache_id in self._cache['batch']:
            batch = self._cache['batch'][cache_id]
        else:
            batch = {
                'input': np.zeros([len(self._input), batch_size, length], dtype=np.int32),
                'len': np.zeros([batch_size], dtype=np.int32),
            }

            if add_gold:
                batch['gold'] = np.zeros([batch_size, length], dtype=np.int32)

            self._cache['batch'][cache_id] = batch

        if not random_batch:
            pos = start_pos

        for i in range(batch_size):
            if random_batch:
                item_length = length
                start = randint(start_pos, total_chars - item_length)
            else:
                start = pos
                item_length = min(length, total_chars - start)

            for j in range(len(self._input)):
                batch['input'][j, i, :item_length] = self._input[j][start:start + item_length]
                if length > item_length:
                    batch['input'][j, i, item_length:] = 0

                #slice = self._input[j][start:start + item_length]
                # print('slice.shape = ' + str(slice.shape))
                #slice = np.pad(slice, (0, length - item_length), 'constant', constant_values=(0, 0))

                #batch['input'][j].append(slice)

            if add_gold:
                batch['gold'][i,:item_length] = self._gold[start:start + item_length]
                if length > item_length:
                    batch['gold'][i, item_length:] = 0
                #gold_slice = self._gold[start:start + item_length]
                #gold_slice = np.pad(gold_slice, (0, length - item_length), 'constant', constant_values=(0, 0))
                #batch['gold'].append(gold_slice)

            batch['len'][i] = item_length

            if not random_batch:
                pos += item_length

        #for j in range(len(batch['input'])):
        #    batch['input'][j] = np.array(batch['input'][j], dtype=np.int32)

        #if add_gold:
        #    batch['gold'] = np.array(batch['gold'], dtype=np.int32)

        batch['len'] = np.array(batch['len'], dtype=np.int32)

        return batch

    @staticmethod
    def _annotate_chars(args, ngrams, chars):
        a = []
        for n in range(len(args['ngrams'])):
            indices = [ ngrams.find(n, '<UNKN>') ] * ( len(chars) - 2 * args['padding_len'] )
            for i in range(args['padding_len'], len(chars) - args['padding_len']):
                start = i + args['ngrams'][n]['start']
                stop = start + args['ngrams'][n]['len']
                s = chars[start:stop]
                idx = ngrams.find(n, s)
                indices[i - args['padding_len']] = idx
            a.append(indices)
        return a

    @staticmethod
    def _generate_tag(args, doc):
        tags = []
        idx = 0

        for sent in doc.sentences():
            for token in sent.tokens():
                while idx < token.start():
                    tags.append('X')
                    idx += 1

                if len(token) == 1:
                    tags.append('S')
                    idx += 1
                else:
                    tags.append('B' + 'I' * (len(token) - 2))
                    tags.append('E')
                    idx += len(token)

            if args['mode'] == 'tokenize&split':
                if tags[-1] == 'S':
                    tags[-1] = 'T'
                else:
                    tags[-1] = 'U'

        tags = ''.join(tags)

        doc_len = len(doc.text())
        while len(tags) < doc_len and doc.text()[idx:idx+1].isspace():
            tags += 'X'
            idx += 1

        if len(doc.text()) != len(tags):
            raise

        return tags

