#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
import time
import json

import tensorflow as tf


def build_fn(config):
    return BiRnnCrf(config)


class BiRnnCrf:
    def __init__(self, config):
        self._config = config
        self._config_saved = False
        self._session = None
        self._build()

    def init(self):
        self._session = tf.compat.v1.Session()
        # config=tf.ConfigProto(
        #    intra_op_parallelism_threads=4,
        #    inter_op_parallelism_threads=4,
        # ))
        self._session.run(tf.compat.v1.global_variables_initializer())

    def train(self, batch, opts):
        loss, accuracy, before = None, None, time.time()

        _, loss, accuracy = self._session.run([
            self._optimizer,
            self._metrics['loss'],
            self._metrics['accuracy']
        ],
            self._prepare_feed_dict(batch, opts)
        )

        return loss, accuracy, time.time() - before

    def evaluate(self, batch):
        loss, accuracy, match, total, before = None, None, None, None, time.time()

        loss, accuracy, match, total = self._session.run([
            self._metrics['loss'],
            self._metrics['accuracy'],
            self._metrics['match'],
            self._metrics['total']
        ],
            self._prepare_feed_dict(batch, {})
        )

        return loss, accuracy, match, total, time.time() - before

    def predict(self, batch):
        loss, accuracy, match, total, before = None, None, None, None, time.time()

        [pred] = self._session.run([
            self._output['predictions']
        ],
            self._prepare_feed_dict(batch, {})
        )

        return pred, time.time() - before

    def config(self):
        return self._config

    def save(self, dir_name, fn_prefix):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        prefix = os.path.join(dir_name, fn_prefix)
        if not self._config_saved:
            self._save_config(prefix + '.conf')
        self._save_model(prefix + '.model')

    def _save_config(self, fn):
        with open(fn, 'w') as f:
            f.write(
                json.dumps(
                    {
                        'conf': {
                            'max_seq_len': self._config['hp']['max_seq_len'],
                            'i2t': self._config['encoding']['tags'],
                            'ngrams': self._config['encoding']['ngrams']
                        },
                        'dicts': {
                            'ngrams': self._config['dicts']['chars'].to_save()
                        }
                    },
                    sort_keys=True,
                    indent=2
                )
            )
        self._config_saved = True

    def _save_model(self, fn):
        # Save the variables as constants
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            self._session,
            tf.compat.v1.get_default_graph().as_graph_def(),
            self._nodes
        )
        with tf.io.gfile.GFile(fn, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

    def load(self, prefix):
        self._load_config(prefix + '.conf')
        self._load_model(prefix + '.model')

    def _load_config(self, fn):
        pass

    def _load_model(self, fn):
        tf.reset_default_graph()

        with tf.gfile.GFile(fn, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        self._restore_input_tensors()
        self._restore_output_tensors()

    def _restore_input_tensors(self):
        self._input = {
                          'idx': [],
                          'len': tf.get_default_graph().get_tensor_by_name('seq_len:0')
                      },

        for i in range(len(self._config['encoding']['ngrams'])):
            t = self._config['encoding']['ngrams'][i]
            idx_name = 'ngram_idx_%d-%d:0' % (t['start'], t['len'])
            tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(idx_name)
            self._input['idx'].append(tensor)

    def _restore_output_tensors(self):
        self._crf = tf.get_default_graph().get_tensor_by_name('CRF/crf:0')
        self._output = {
            'predictions': tf.get_default_graph().get_tensor_by_name('Dense/dropout/Identity:0')
        }

    def _prepare_feed_dict(self, batch, opts):
        feed_dict = {}

        for opt in opts:
            feed_dict[self._input[opt]] = opts[opt]

        for i in range(len(self._input['idx'])):
            feed_dict[self._input['idx'][i]] = batch['input'][i]

        feed_dict[self._input['seq_len']] = batch['len']

        if 'gold' in batch:
            feed_dict[self._input['gold']] = batch['gold']

        return feed_dict

    def _build(self):
        tf.compat.v1.reset_default_graph()

        self._session = None

        self._input = {
            'input_keep_prob': tf.compat.v1.placeholder_with_default(1.0, [], name='input_keep_prob'),
            'rnn_output_keep_prob': tf.compat.v1.placeholder_with_default(1.0, [], name='rnn_output_keep_prob'),
            'dense_keep_prob': tf.compat.v1.placeholder_with_default(1.0, [], name='dense_keep_prob'),
            'learning_rate': tf.compat.v1.placeholder_with_default(0.1, [], name='learning_rate'),

            'idx': [],
            'seq_len': tf.compat.v1.placeholder(tf.int32, [None], name="seq_len"),
            'gold': tf.compat.v1.placeholder(tf.int32, [None, self._config['hp']['max_seq_len']], name="gold")
        }
        self._metrics = {}
        self._output = {}
        self._crf = None
        self._dense_output = None

        mask = tf.sequence_mask(self._input['seq_len'], self._config['hp']['max_seq_len'])
        total_words = tf.reduce_sum(self._input['seq_len'])

        all_inputs = []
        embd_reg_losses = []

        dicts = self._config['dicts']['chars']

        for i in range(len(self._config['encoding']['ngrams'])):
            t = self._config['encoding']['ngrams'][i]

            idx = tf.compat.v1.placeholder(tf.int32,
                                           shape=[None, self._config['hp']['max_seq_len']],
                                           name='ngram_idx_%d-%d' % (t['start'], t['len']))
            self._input['idx'].append(idx)

            embd = tf.compat.v1.get_variable(name='ngram_embd_%d-%d' % (t['start'], t['len']),
                                             shape=[dicts.size(i), t['embd_size']],
                                             initializer=tf.contrib.layers.xavier_initializer()
                                             )

            input_trainable_ngram_ids, _ = tf.unique(tf.reshape(idx, [-1]))
            print("input_trainable_ngram_ids.shape = " + str(input_trainable_ngram_ids.shape))
            input_trainable_ngram_embd_slice = tf.gather(embd, input_trainable_ngram_ids)
            print("input_trainable_ngram_embd_slice.shape = " + str(input_trainable_ngram_embd_slice.shape))
            input_trainable_ngram_reg_loss = tf.nn.l2_loss(input_trainable_ngram_embd_slice) * self._config['hp'][
                'embd_l2']
            embd_reg_losses.append(input_trainable_ngram_reg_loss)

            input_name = 'ngram_input_%d-%d' % (t['start'], t['len'])
            input = tf.nn.embedding_lookup(name=input_name,
                                           params=embd,
                                           ids=idx)

            all_inputs.append(input)

        input = tf.concat(all_inputs, 2)
        print('input.shape = ' + str(input.shape))

        input = tf.layers.dropout(input, self._input['input_keep_prob'])

        rnn_output = self._build_fused_rnn('BiRNN', self._config, input, self._input['seq_len'])

        print('rnn_output.shape = ' + str(rnn_output.shape))

        rnn_output = tf.layers.dropout(rnn_output, self._input['rnn_output_keep_prob'])

        with tf.compat.v1.variable_scope('Dense'):
            dense_input = rnn_output
            dense_output = dense_input  # in case of absense of dense layers

            for n in range(len(self._config['model']['dense'])):
                layer = self._config['model']['dense'][n]

                dense_output = tf.layers.dense(dense_input,
                                               layer['dim'],
                                               #activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer()
                                               )

                dense_output = tf.layers.dropout(dense_output,
                                                 self._input['dense_keep_prob']
                                                 )

                dense_input = dense_output

        self._dense_output = dense_output

        print('dense_output.shape = ' + str(self._dense_output.shape))

        with tf.variable_scope('CRF'):
            self._crf = tf.compat.v1.get_variable("crf",
                                                  [len(self._config['encoding']['i2t']),
                                                   len(self._config['encoding']['i2t'])],
                                                  dtype=tf.float32
                                                  )

            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(dense_output,
                                                                  self._input['gold'],
                                                                  self._input['seq_len'],
                                                                  self._crf
                                                                  )

        # optimizer
        self._metrics['loss'] = tf.reduce_mean(-log_likelihood) + tf.reduce_sum(embd_reg_losses)
        print('loss.shape = ' + str(self._metrics['loss'].shape))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self._input['learning_rate'], beta2=0.9)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)

        self._optimizer = optimizer.minimize(self._metrics['loss'],
                                             global_step=tf.compat.v1.train.get_or_create_global_step())

        # gradient_var_pairs = optimizer.compute_gradients(loss) #, tf.trainable_variables())
        # vars = [x[1] for x in gradient_var_pairs]
        # gradients = [x[0] for x in gradient_var_pairs]
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        # train_step = optimizer.apply_gradients(zip(clipped_gradients, vars))
        # optimizer = train_step

        # predictions
        self._output['predictions'], _ = tf.contrib.crf.crf_decode(dense_output,
                                                                   self._crf,
                                                                   self._input['seq_len']
                                                                   )

        # metrics
        self._metrics['match'] = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(self._output['predictions'], self._input['gold']), mask), tf.int32))
        self._metrics['accuracy'] = tf.truediv(self._metrics['match'], total_words)
        self._metrics['total'] = total_words

        self._nodes = [self._dense_output.name.split(':')[0], self._crf.name.split(':')[0]]

    @staticmethod
    def _build_fused_rnn(scope, conf, input, seq_len):
        with tf.compat.v1.variable_scope(scope):
            input = tf.contrib.rnn.transpose_batch_time(input)

            for n in range(len(conf['model']['rnn'])):
                l = conf['model']['rnn'][n]

                if l['type'] == 'lstm':
                    fw_rnn_cell = tf.contrib.rnn.LSTMBlockFusedCell(l['fw_dim'])
                    bw_rnn_cell = tf.contrib.rnn.LSTMBlockFusedCell(l['bw_dim'])
                elif l['type'] == 'gru':
                    fw_rnn_cell = tf.contrib.rnn.GRUBlockCellV2(l['fw_dim'])
                    bw_rnn_cell = tf.contrib.rnn.GRUBlockCellV2(l['bw_dim'])
                else:
                    sys.stderr.write('ERROR: unknown rnn cell type \'%s\'' % l['type'])
                    return None

                fw_output, _ = fw_rnn_cell(input, dtype=tf.float32)
                bw_input = tf.reverse_sequence(input, seq_len, seq_axis=0, batch_axis=1)
                bw_output, _ = bw_rnn_cell(bw_input, dtype=tf.float32)
                bw_output = tf.reverse_sequence(bw_output, seq_len, seq_axis=0, batch_axis=1)

                rnn_output = tf.concat([fw_output, bw_output], 2)
                input = rnn_output

            rnn_output = tf.contrib.rnn.transpose_batch_time(rnn_output)
            return rnn_output

    def print_model_statistics(self):
        for layer in self._config['model']['rnn']:
            print('rnn layer dim = %d' % layer['fw_dim'])
        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print('%s\t%s\t%d' % (variable.name, str(shape), variable_parameters))
            total_parameters += variable_parameters
        print(total_parameters)
