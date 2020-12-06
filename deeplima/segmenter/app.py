#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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
