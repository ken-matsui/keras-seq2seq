# coding: utf-8

import os
# from glob import glob
# from os.path import relpath, splitext

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *
import numpy as np

from model import AttSeq2Seq


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
    try: os.makedirs(FLAGS.out)
    except: pass

    seq2seq = AttSeq2Seq(embed_size=FLAGS.embed_size,
                         data_dir=FLAGS.data_dir,
                         decode_max_size=FLAGS.decode_max_size)
    seq2seq.train(batch_size=FLAGS.batch_size,
                  embed_size=FLAGS.embed_size,
                  epochs=FLAGS.epoch,
                  output_path=FLAGS.model_dir)

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
        target_seq = np.zeros((1, 1, seq2seq.num_vocabularies))
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
            target_seq = np.zeros((1, 1, seq2seq.num_vocabularies))
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
