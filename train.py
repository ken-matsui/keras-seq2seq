# coding: utf-8

import os
# from glob import glob
# from os.path import relpath, splitext

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np

from model import AttSeq2Seq


flags = tf.flags
flags.DEFINE_integer('epochs', 100, "Number of epochs")
flags.DEFINE_bool('resume', False, "Resume mode if this flag is set")
flags.DEFINE_integer('batch_size', 64, "Number of batch size")
flags.DEFINE_integer('embed_size', 256, "Number of embed(vector) size")
flags.DEFINE_integer('decode_max_size', 15,
                     """Decoding processing is terminated when EOS did output,
                        but output the maximum vocabulary number when it did not output""")
flags.DEFINE_string('vocab_file', './data/vocab.txt', "Directory to vocab file")
flags.DEFINE_string('infile', './data/dataid.txt', "Directory to id file")
flags.DEFINE_string('out', './models', "Directory to output the result")
flags.DEFINE_integer('select', 0, "Select npz file.")
flags.DEFINE_bool('gpu', False, 'GPU mode if this flag is set')
FLAGS = flags.FLAGS


def load_vocab():
    # Load word dictionary
    with open(FLAGS.vocab_file, 'r') as f:
        lines = f.readlines()
    return list(map(lambda s: s.replace("\n", ""), lines))


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


def main(_):
    input_texts, target_texts = load_ids()
    vocab = load_vocab()
    num_vocabularies = len(vocab)

    print('Number of samples:', len(input_texts))
    print('Number of unique vocabularies:', num_vocabularies)
    print('Max sequence length:', FLAGS.decode_max_size)

    encoder_input_data = np.zeros(
        (len(input_texts), FLAGS.decode_max_size, num_vocabularies),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), FLAGS.decode_max_size, num_vocabularies),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), FLAGS.decode_max_size, num_vocabularies),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, char] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, char] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, char] = 1.

    print('GPU:', FLAGS.gpu)
    print('Minibatch size:', FLAGS.batch_size)
    print('embed_size:', FLAGS.embed_size)
    print('epoch:', FLAGS.epochs)
    print()

    try:
        os.makedirs(FLAGS.out)
    except:
        pass

    seq2seq = AttSeq2Seq(FLAGS.embed_size, num_vocabularies)
    model = Model(inputs=[seq2seq.encoder_inputs, seq2seq.decoder_inputs],
                  outputs=seq2seq.decoder_outputs)
    plot_model(model, to_file='model.png')
    chkpt_path = os.path.join(FLAGS.out, "weights.{epoch:02d}.hdf5")
    chkpt = ModelCheckpoint(chkpt_path, period=1)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epochs,
              callbacks=[chkpt])
    # Save model
    model.save(os.path.join(FLAGS.out, str(FLAGS.epochs) + '.h5'))

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(seq2seq.encoder_inputs, seq2seq.encoder_states)

    decoder_state_input_h = Input(shape=(FLAGS.embed_size,))
    decoder_state_input_c = Input(shape=(FLAGS.embed_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = seq2seq.decoder(
        seq2seq.decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = seq2seq.decoder_dense(decoder_outputs)
    decoder_model = Model(
        [seq2seq.decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    # reverse_input_char_index = dict(
    #     (i, char) for char, i in input_token_index.items())
    # reverse_target_char_index = dict(
    #     (i, char) for char, i in target_token_index.items())

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_vocabularies))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, vocab.index('\t')] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = vocab[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > FLAGS.decode_max_size):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_vocabularies))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
    tf.app.run()
