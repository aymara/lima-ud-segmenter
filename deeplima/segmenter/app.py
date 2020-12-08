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

import argparse

from deeplima.segmenter.train import train


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('-t', '--treebank', help='Treebank name')
    parser.add_argument('-u', '--ud-path', help='Path to Universal Dependencies treebanks')
    parser.add_argument('-c', '--config', help='Configuration name', default='default')
    parser.add_argument('-p', '--guess-hp', action='store_true', help='Guess hyperparameters', default=False)
    parser.add_argument('-l', '--iter-len', help='Length of one iteration', type=int, default=50)
    parser.add_argument('-m', '--max-iter-wo-improvement', help='Max iterations without dev accuracy improvement',
                        type=int, default=100)
    parser.add_argument('-o', '--output-dir', help='Output directory')
    parser.add_argument('-f', '--file-prefix', help='Prefix for output files', default='segmenter')

    args = parser.parse_args()

    if args.action == 'train':
        train(args)
    elif args.action == 'predict':
        pass
    else:
        parser.print_help()


if __name__ == '__main__':
    run()
