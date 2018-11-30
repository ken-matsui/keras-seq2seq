# coding: utf-8

# import os
# from glob import glob
# from os.path import relpath, splitext

import tensorflow as tf
# from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend
import numpy as np

from model import AttSeq2Seq


flags = tf.flags
flags.DEFINE_integer('epochs', 200, "Number of epochs")
flags.DEFINE_bool('resume', False, "Resume mode if this flag is set")
flags.DEFINE_integer('batch_size', 20, "Number of batch size")
flags.DEFINE_integer('embed_size', 100, "Number of embed(vector) size")
flags.DEFINE_integer('hidden_size', 100, "Number of hidden units")
# デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
flags.DEFINE_integer('decode_max_size', 15, "Number of decode max size")
flags.DEFINE_string('vocab_file', './data/vocab.txt', "Directory to vocab file")
flags.DEFINE_string('infile', './data/dataid.txt', "Directory to id file")
flags.DEFINE_string('out', './result/', "Directory to output the result")
flags.DEFINE_integer('select', 0, "Select npz file.")
flags.DEFINE_bool('gpu', False, 'GPU mode if this flag is set')
FLAGS = flags.FLAGS


def load_vocab():
    # 単語辞書データを取り出す
    with open(FLAGS.vocab_file, 'r') as f:
        lines = f.readlines()
    return list(map(lambda s: s.replace("\n", ""), lines))


def load_ids():
    # 対話データ(ID版)を取り出す
    queries, responses = [], []
    with open(FLAGS.infile, 'r') as f:
        for l in f.read().split('\n')[:-1]:
            # queryとresponseで分割する
            d = l.split('\t')
            # ミニバッチ対応のため，単語数サイズを調整してNumpy変換する
            queries.append(batch_ids(list(map(int, d[0].split(',')[:-1])), "query"))
            responses.append(batch_ids(list(map(int, d[1].split(',')[:-1])), "response"))
    return queries, responses


def batch_ids(ids, sentence_type):  # TODO: 0を補填する (<eos>)
    if sentence_type == "query":  # queryの場合は前方に-1を補填する
        if len(ids) > FLAGS.decode_max_size:  # ミニバッチ単語サイズになるように先頭から削る
            del ids[0:len(ids) - FLAGS.decode_max_size]
        else:  # ミニバッチ単語サイズになるように前方に付け足す
            # ids = ([-1] * (FLAGS.decode_max_size - len(ids))) + ids
            ids = ([0] * (FLAGS.decode_max_size - len(ids))) + ids
    elif sentence_type == "response":  # responseの場合は後方に-1を補填する
        if len(ids) > FLAGS.decode_max_size:  # ミニバッチ単語サイズになるように末尾から削る
            del ids[FLAGS.decode_max_size:]
        else:  # ミニバッチ単語サイズになるように後方に付け足す
            # ids = ids + ([-1] * (FLAGS.decode_max_size - len(ids)))
            ids = ids + ([0] * (FLAGS.decode_max_size - len(ids)))
    return ids


def main(_):
    print('GPU: {}'.format(FLAGS.gpu))
    print('# Minibatch-size: {}'.format(FLAGS.batch_size))
    print('# embed_size: {}'.format(FLAGS.embed_size))
    print('# hidden_size: {}'.format(FLAGS.hidden_size))
    print('# epoch: {}'.format(FLAGS.epochs))
    print()

    # 単語辞書の読み込み
    vocab = load_vocab()
    seq2seq = AttSeq2Seq(vocab_size=len(vocab),
                         embed_size=FLAGS.embed_size,
                         hidden_size=FLAGS.hidden_size,
                         batch_size=FLAGS.batch_size)

    inputs = seq2seq.inputs
    outputs = seq2seq.decode(backend.zeros((FLAGS.batch_size,)))

    # Load train data
    queries, responses = load_ids()
    train_queries = np.array(queries)
    train_responses = np.array(responses)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_queries,
              train_responses,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs)

    model.save(FLAGS.out + str(FLAGS.epoch) + '.h5')


if __name__ == '__main__':
    tf.app.run()
