# coding: utf-8

import os

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.activations import tanh
from tensorflow.python.keras.backend import zeros, exp, batch_dot
# from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import plot_model
import numpy as np

__all__ = ['AttSeq2Seq']


class AttLSTMDecoder(object):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        Attention Model + LSTM Decoder
        :param vocab_size: 語彙数
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(AttLSTMDecoder, self).__init__()
        # 単語を単語ベクトルに変換する層
        self.ye = Embedding(vocab_size, embed_size, mask_zero=True)
        # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
        self.eh = Dense(4 * hidden_size, input_shape=(embed_size,))
        # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
        self.hh = Dense(4 * hidden_size, input_shape=(hidden_size,))
        # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
        self.fh = Dense(4 * hidden_size, input_shape=(hidden_size,))
        # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
        self.bh = Dense(4 * hidden_size, input_shape=(hidden_size,))
        # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
        self.he = Dense(embed_size, input_shape=(hidden_size,))
        # 単語ベクトルを語彙数サイズのベクトルに変換する層
        self.ey = Dense(vocab_size, input_shape=(embed_size,))

    def __call__(self, y, c, h, f, b):
        """
        Decoderの計算
        :param y: Decoderに入力する単語
        :param c: 内部メモリ
        :param h: Decoderの中間ベクトル
        :param f: Attention Modelで計算された順向きEncoderの加重平均
        :param b: Attention Modelで計算された逆向きEncoderの加重平均
        :return: 語彙数サイズのベクトル、更新された内部メモリ、更新された中間ベクトル
        """
        with tf.variable_scope("AttLSTMDecoder"):
            # 単語を単語ベクトルに変換
            e = tanh(self.ye(y))
            # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
            e = Add()([self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b)])
            c, h = LSTM(c)(e)
            # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
            t = self.ey(tanh(self.he(h)))
        return t, c, h


class Attention(object):
    def __init__(self, hidden_size, batch_size):
        """
        Attentionのインスタンス化
        :param hidden_size: 隠れ層のサイズ
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # 順向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
        self.fh = Dense(hidden_size, input_shape=(hidden_size,))
        # 逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
        self.bh = Dense(hidden_size, input_shape=(hidden_size,))
        # Decoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
        self.hh = Dense(hidden_size, input_shape=(hidden_size,))
        # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
        self.hw = Dense(1, input_shape=(hidden_size,))

    def __call__(self, fs, bs, h):
        """
        Attentionの計算
        :param fs: 順向きのEncoderの中間ベクトルが記録されたリスト
        :param bs: 逆向きのEncoderの中間ベクトルが記録されたリスト
        :param h: Decoderで出力された中間ベクトル
        :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        """
        # weight
        ws = []
        sum_w = zeros((self.batch_size, 1))  # ウェイトの合計値を計算するための値を初期化
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = tanh(self.fh(f) + self.bh(b) + self.hh(h))
            w = exp(self.hw(w))  # softmax関数を使って正規化する
            ws.append(w)  # 計算したウェイトを記録
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = zeros((self.batch_size, self.hidden_size))
        att_b = zeros((self.batch_size, self.hidden_size))
        for f, b, w in zip(fs, bs, ws):
            w /= sum_w  # ウェイトの和が1になるように正規化
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += Reshape((self.batch_size, self.hidden_size))(batch_dot(f, w))
            att_b += Reshape((self.batch_size, self.hidden_size))(batch_dot(b, w))
        return att_f, att_b


class AttSeq2Seq(object):
    def __init__(self, embed_size, data_dir, decode_max_size):
        """
        Sequence to Sequence with Attention Model
        """
        super(AttSeq2Seq, self).__init__()
        # Variables
        self.num_vocabularies = len(self.__load_vocabularies(data_dir))
        self.data_dir = data_dir
        self.decode_max_size = decode_max_size

        # Layers
        self.encoder_inputs = Input(shape=(None,))
        encoder_embed = Embedding(self.num_vocabularies, embed_size)
        encoder = LSTM(embed_size, return_state=True)

        self.decoder_inputs = Input(shape=(None,))
        decoder_embed = Embedding(self.num_vocabularies, embed_size)  # mask_zero=True
        self.decoder = LSTM(embed_size, return_sequences=True, return_state=True)
        self.decoder_dense = Dense(self.num_vocabularies, activation='softmax')

        # Calculation Graph
        with tf.variable_scope("AttSeq2Seq"):
            x = encoder_embed(self.encoder_inputs)
            encoder_outputs, state_h, state_c = encoder(x)
            # We discard `encoder_outputs` and only keep the states.
            self.encoder_states = [state_h, state_c]

            x = decoder_embed(self.decoder_inputs)
            decoder_outputs, _, _ = self.decoder(x, initial_state=self.encoder_states)
            self.decoder_outputs = self.decoder_dense(decoder_outputs)

    def train(self, batch_size, embed_size, epochs, output_path):
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
        print('Size of mini batch:', batch_size)
        print('Size of embed:', embed_size)
        print('Number of epochs:', epochs)
        print()

        model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        plot_model(model, to_file='model.png', show_shapes=True)
        # chkpt_path = os.path.join(output_path, "weights.{epoch:02d}.hdf5")
        # chkpt = ModelCheckpoint(chkpt_path, period=1)
        model.summary()
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)
        # Save model
        model.save(os.path.join(output_path, str(epochs) + '.h5'))

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
        # queries = np.empty((0, self.decode_max_size, 1), int)
        queries = np.empty((0, self.decode_max_size), int)
        # responses = np.empty((0, self.decode_max_size, 1), int)    # [[[2], [3], [2]],
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
