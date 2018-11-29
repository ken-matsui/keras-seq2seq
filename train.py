# coding: utf-8

import os
from glob import glob
from os.path import relpath, splitext

import tensorflow as tf

from AttSeq2Seq.model import AttSeq2Seq
from AttSeq2Seq.trainer import Trainer


flags = tf.flags
flags.DEFINE_integer('-e', '--epoch', 200, "Number of epoch")
flags.DEFINE_bool('-r', '--resume', False, "Resume mode if this flag is set")
flags.DEFINE_integer('-b', '--batchsize', 20, "Number of batch size")
flags.DEFINE_integer('-es', '--embed_size', 100, "Number of embed(vector) size")
flags.DEFINE_integer('-n', '--n_hidden', 100, "Number of hidden units")
# デコードはEOSが出力されれば終了する、出力されない場合の最大出力語彙数
flags.DEFINE_integer('-d', '--decode_max_size', 15, "Number of decode max size")
flags.DEFINE_string('-v', '--vocab_file', './data/vocab.txt', "Directory to vocab file")
flags.DEFINE_string('-i', '--infile', './data/dataid.txt', "Directory to id file")
flags.DEFINE_string('-o', '--out', './result/', "Directory to output the result")
flags.DEFINE_integer('-s', '--select', 0, "Select npz file.")
flags.DEFINE_bool('-g', '--gpu', False, 'GPU mode if this flag is set')
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


def batch_ids(ids, sentence_type):
    if sentence_type == "query":  # queryの場合は前方に-1を補填する
        if len(ids) > FLAGS.decode_max_size:  # ミニバッチ単語サイズになるように先頭から削る
            del ids[0:len(ids) - FLAGS.decode_max_size]
        else:  # ミニバッチ単語サイズになるように前方に付け足す
            ids = ([-1] * (FLAGS.decode_max_size - len(ids))) + ids
    elif sentence_type == "response":  # responseの場合は後方に-1を補填する
        if len(ids) > FLAGS.decode_max_size:  # ミニバッチ単語サイズになるように末尾から削る
            del ids[FLAGS.decode_max_size:]
        else:  # ミニバッチ単語サイズになるように後方に付け足す
            ids = ids + ([-1] * (FLAGS.decode_max_size - len(ids)))
    return ids


def main():
    print('GPU: {}'.format(FLAGS.gpu))
    print('# Minibatch-size: {}'.format(FLAGS.batchsize))
    print('# embed_size: {}'.format(FLAGS.embed_size))
    print('# n_hidden: {}'.format(FLAGS.n_hidden))
    print('# epoch: {}'.format(FLAGS.epoch))
    print()

    # 単語辞書の読み込み
    vocab = load_vocab()
    model = AttSeq2Seq(vocab_size=len(vocab),
                       embed_size=FLAGS.embed_size,
                       hidden_size=FLAGS.n_hidden)
    # 学習用データを読み込む
    queries, responses = load_ids()
    if FLAGS.resume:
        if FLAGS.select == 0:
            # 最新のモデルデータを使用する．
            files = [splitext(relpath(s, FLAGS.out))[0] for s in glob(FLAGS.out + "*.npz")]
            num = max(list(map(int, files)))
        else:
            # 指定のモデルデータを使用する．
            num = FLAGS.select
        npz = FLAGS.out + str(num) + ".npz"
        print("Resume training from", npz)
    else:
        try:
            os.mkdir(FLAGS.out)
        except:
            pass
        print("Train")
        npz = None
    trainer = Trainer(model, npz)
    trainer.fit(queries=queries,
                responses=responses,
                train_path=FLAGS.out,
                epoch_num=FLAGS.epoch,
                batch_size=FLAGS.batchsize)


if __name__ == '__main__':
    tf.app.run()
