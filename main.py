# coding: utf-8

import os
# from glob import glob
# from os.path import relpath, splitext

import tensorflow as tf

from model import Seq2Seq


flags = tf.flags
flags.DEFINE_string('phase',       'train',  "train or test")
flags.DEFINE_string('data_dir',    'data',   "Data directory")
flags.DEFINE_string('model_dir',   'models', "Directory to output the result")
flags.DEFINE_integer('epoch',      100,      "Number of epochs")
flags.DEFINE_integer('batch_size', 64,       "Number of batch size")
flags.DEFINE_integer('embed_size', 256,      "Number of embed(word vector) size")
flags.DEFINE_integer('decode_max_size', 15,
                     """Decoding processing is terminated when EOS did output,
                        but output the maximum vocabulary number when it did not output""")
FLAGS = flags.FLAGS


def main(_):
    try: os.makedirs(FLAGS.model_dir)
    except: pass

    seq2seq = Seq2Seq(embed_size=FLAGS.embed_size,
                      data_dir=FLAGS.data_dir,
                      decode_max_size=FLAGS.decode_max_size,
                      batch_size=FLAGS.batch_size,
                      epochs=FLAGS.epoch,
                      output_path=FLAGS.model_dir)
    seq2seq.train()
    seq2seq.inference()


if __name__ == '__main__':
    tf.app.run()
