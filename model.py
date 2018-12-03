# coding: utf-8

import os

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import plot_model
import numpy as np

__all__ = ['Seq2Seq']


class Seq2Seq(object):
    def __init__(self, embed_size, data_dir, decode_max_size, batch_size, epochs, output_path):
        """
        Sequence to Sequence with Attention Model
        """
        super(Seq2Seq, self).__init__()
        # Variables
        self.embed_size = embed_size
        self.num_vocabularies = len(self.__load_vocabularies(data_dir))
        self.data_dir = data_dir
        self.decode_max_size = decode_max_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_path = output_path

        # Layers
        self.encoder_inputs = Input(shape=(None,))  # None == batch_size (when data_num % batch_size == 0)
        encoder_embed = Embedding(self.num_vocabularies, embed_size)
        encoder = LSTM(embed_size, return_state=True)

        self.decoder_inputs = Input(shape=(None,))
        self.decoder_embed = Embedding(self.num_vocabularies, embed_size)  # mask_zero=True
        self.decoder = LSTM(embed_size, return_sequences=True, return_state=True)
        self.decoder_dense = Dense(self.num_vocabularies, activation='softmax')

        # Calculation Graph
        with tf.variable_scope("AttSeq2Seq"):
            x = encoder_embed(self.encoder_inputs)
            encoder_outputs, state_h, state_c = encoder(x)
            # We discard `encoder_outputs` and only keep the states.
            self.encoder_states = [state_h, state_c]

            x = self.decoder_embed(self.decoder_inputs)
            decoder_outputs, _, _ = self.decoder(x, initial_state=self.encoder_states)
            self.decoder_outputs = self.decoder_dense(decoder_outputs)

    def train(self):
        # Load integer sequences
        encoder_input_data, decoder_input_data = self.__load_ids()

        # Note that `decoder_target_data` needs to be one-hot encoded,
        # rather than sequences of integers like `decoder_input_data`!
        decoder_target_data = np.zeros(
            (len(encoder_input_data), self.decode_max_size, self.num_vocabularies),
            dtype='float32')
        for i, target_text in enumerate(decoder_input_data):
            for t, char in enumerate(target_text):
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, char] = 1.

        print('Number of samples:', len(encoder_input_data))
        print('Number of unique vocabularies:', self.num_vocabularies)
        print('Max sequence length:', self.decode_max_size)
        print('Size of mini batch:', self.batch_size)
        print('Size of embed:', self.embed_size)
        print('Number of epochs:', self.epochs)
        print()

        model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        plot_model(model, to_file='model.png', show_shapes=True)

        callbacks = [TensorBoard(batch_size=self.batch_size)]
        # ckpt_path = os.path.join(output_path, "weights.{epoch:02d}.hdf5")
        # ckpt = ModelCheckpoint(chkpt_path, period=1)

        model.summary()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.2,
                  callbacks=callbacks)
        # Save model
        model.save(os.path.join(self.output_path, str(self.epochs) + '.h5'))

    @staticmethod
    def __load_vocabularies(data_dir):
        """
        Load word dictionary
        """
        with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
            lines = f.readlines()
        return list(map(lambda s: s.replace("\n", ""), lines))

    def __load_ids(self):
        """
        Load dialogue data (ID)
        :return Tuple(ndarray, ndarray):
        """
        queries = np.empty((0, self.decode_max_size), int)
        responses = np.empty((0, self.decode_max_size), int)
        with open(os.path.join(self.data_dir, 'dataid.txt'), 'r') as f:
            for l in f.read().split('\n')[:-1]:
                # Split by query and response
                d = l.split('\t')
                # Adjust the word size for mini batch compatibility
                qs = self.__batch_ids(list(map(int, d[0].split(',')[:-1])), "query")
                queries = np.append(queries, np.array([qs]), axis=0)
                rs = self.__batch_ids(list(map(int, d[1].split(',')[:-1])), "response")
                responses = np.append(responses, np.array([rs]), axis=0)
        return queries, responses

    def __batch_ids(self, ids, sentence_type):
        if sentence_type == "query":  # queryの場合は前方に-1を補填する
            if len(ids) > self.decode_max_size:  # ミニバッチ単語サイズになるように先頭から削る
                del ids[0:len(ids) - self.decode_max_size]
            else:  # ミニバッチ単語サイズになるように前方に付け足す 0 -> (<eos>)
                # ids = ([-1] * (FLAGS.decode_max_size - len(ids))) + ids
                ids = ([0] * (self.decode_max_size - len(ids))) + ids
        elif sentence_type == "response":  # responseの場合は後方に-1を補填する
            if len(ids) > self.decode_max_size:  # ミニバッチ単語サイズになるように末尾から削る
                del ids[self.decode_max_size:]
            else:  # ミニバッチ単語サイズになるように後方に付け足す
                # ids = ids + ([-1] * (FLAGS.decode_max_size - len(ids)))
                ids = ids + ([0] * (self.decode_max_size - len(ids)))
        return ids

    def inference(self):
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.embed_size,))
        decoder_state_input_c = Input(shape=(self.embed_size,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder(
            self.decoder_embed(self.decoder_inputs), initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        # reverse_input_char_index = dict(
        #     (i, char) for char, i in input_token_index.items())
        # reverse_target_char_index = dict(
        #     (i, char) for char, i in target_token_index.items())

        vocab = self.__load_vocabularies(self.data_dir)

        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, self.num_vocabularies))

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            decoded_sentence = ''
            for i in range(self.decode_max_size):
                output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value)

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = vocab[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if sampled_char == "<eos>":
                    break

                # Update the target sequence (of length 1).
                # target_seq = np.zeros((1, self.num_vocabularies))
                # target_seq[0, sampled_token_index] = 1.
                # target_seq = np.zeros((1, self.num_vocabularies))
                target_seq[0, i] = sampled_token_index

                # Update states
                states_value = [h, c]

            return decoded_sentence

        # Load integer sequences
        encoder_input_data, decoder_input_data = self.__load_ids()
        for i in range(100):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = encoder_input_data[i]
            decoded_sentence = decode_sequence(input_seq)
            print('-')
            print('Input sentence:', ''.join(list(map(lambda x: vocab[x], encoder_input_data[i]))))
            print('Decoded sentence:', decoded_sentence)
