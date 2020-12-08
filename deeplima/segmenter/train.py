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

import sys
import os
import time

from datetime import datetime

from deeplima.io.conllu.treebank import Treebank
from deeplima.segmenter.data.dataset import Dataset

from deeplima.segmenter.config.config import get_config
from deeplima.segmenter.config.hp import guess_hp
from deeplima.segmenter.models.models import get_model


def train(args):
    # Default output files
    if args.output_dir is None:
        args.output_dir = args.treebank + '@' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Preparing config
    config = get_config(args.config)
    tb = Treebank(args.ud_path, args.treebank)
    train_set = Dataset(config['encoding'], tb.set('train'))
    if args.guess_hp:
        guess_hp(args.config, config, train_set)
    config['dicts']['chars'] = train_set.ngrams()

    # Loading dev_set
    dev_set = Dataset(config['encoding'], tb.set('dev'), config['dicts']['chars'])

    # Building model
    build_model_fn = get_model(config['model']['name'])
    model = build_model_fn(config)

    model.init()
    lr = config['hp']['learning_rate']

    history = {'train': {'loss': [], 'accuracy': []}, 'dev': {'loss': [], 'accuracy': []}}
    iter_no = 0
    best_accuracy = 0
    best_iter = 0

    model.print_model_statistics()

    # Training
    while True:
        metrics = {'loss': [0] * args.iter_len, 'accuracy': [0] * args.iter_len}
        before = time.time()
        for i in range(args.iter_len):
            batch = train_set.training_batch(config['hp']['max_seq_len'], config['hp']['batch_size'])
            # sys.stderr.write('Batch generation: %.4f\n' % (time.time() - before))
            loss, accuracy, duration = model.train(batch, {
                'learning_rate': lr,
                'input_keep_prob': config['hp']['input_keep_prob'],
                'rnn_output_keep_prob': config['hp']['rnn_output_keep_prob'],
                'dense_keep_prob': config['hp']['dense_keep_prob'],
            })
            metrics['loss'][i] = loss
            metrics['accuracy'][i] = accuracy

        duration = time.time() - before
        loss = float(sum(metrics['loss'])) / int(args.iter_len)
        accuracy = float(sum(metrics['accuracy'])) / int(args.iter_len)

        history['train']['loss'].append(loss)
        history['train']['accuracy'].append(accuracy)

        # Evaluation
        dev_loss, dev_accuracy, eval_duration = evaluate_model(model, dev_set)

        history['dev']['loss'].append(dev_loss)
        history['dev']['accuracy'].append(dev_accuracy)

        mark = ' '
        if dev_accuracy > best_accuracy:
            best_iter, best_accuracy, mark = iter_no, dev_accuracy, '*'
            model.save(args.output_dir, args.file_prefix)

        print(
            '%4d TRAIN lr=%.6f loss=%8.4f acc=%.6f time=%.2f | DEV loss=%8.4f acc=%.6f %s time=%.2f | TOTAL TIME %.2f' % (
                iter_no, lr, loss, accuracy, duration, dev_loss, dev_accuracy, mark, eval_duration,
                time.time() - before))

        if len(history['train']['loss']) > 2:
            impr = (history['train']['loss'][-1] - history['train']['loss'][-2]) / history['train']['loss'][-1]
            if impr > -0.05 and (iter_no - best_iter) > 2:
                lr = lr * (1.0 - config['hp']['learning_rate_decay'])

        if iter_no - best_iter >= args.max_iter_wo_improvement or lr < 0.0001:
            break

        iter_no += 1

    print('Summary:\nBest dev accuracy = %.6f on iter #%d' % (best_accuracy, best_iter))


def evaluate_model(model, ds, outside_of_tf = False):
    loss = []
    match = 0
    total = 0
    before = time.time()

    if outside_of_tf:
        for batch in ds.prediction_batch(model.config()['hp']['max_seq_len'], model.config()['hp']['batch_size'], 0):
            p, d = model.predict(batch)
            for i in range(p.shape[0]):
                for j in range(batch['len'][i]):
                    if batch['gold'][i,j] == p[i,j]:
                        match += 1
                    total += 1
    else:
        for batch in ds.prediction_batch(model.config()['hp']['max_seq_len'], model.config()['hp']['batch_size'], 0):
            l, a, m, t, d = model.evaluate(batch)
            loss.append(l)
            match += m
            total += t

    avg_loss = 0 if len(loss) == 0 else float(sum(loss)) / len(loss)
    return avg_loss, float(match) / total, time.time() - before
