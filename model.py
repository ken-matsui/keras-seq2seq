# coding: utf-8

import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
import numpy as np

__all__ = ['AttSeq2Seq']


class LSTMEncoder(object):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        LSTM Encoder
        :param vocab_size: 使われる単語の種類数
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(LSTMEncoder, self).__init__()
        # self.xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1)
        self.xe = Embedding(vocab_size, embed_size, mask_zero=True)
        # self.eh = L.Linear(embed_size, 4 * hidden_size)
        self.eh = Dense(4 * hidden_size, input_shape=(embed_size,))
        # self.hh = L.Linear(hidden_size, 4 * hidden_size)
        self.hh = Dense(4 * hidden_size, input_shape=(hidden_size,))

    def __call__(self, x, c, h):
        """
        Encoderの計算
        :param x: one-hotな単語
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        with tf.variable_scope("LSTMEncoder"):
            e = self.xe(x)
            e = activations.tanh(e)
            e = LSTM(c)(self.eh(e) + self.hh(h))
        return e


class AttLSTMDecoder(object):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        Attention Model + LSTM Decoder
        :param vocab_size: 語彙数
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(AttLSTMDecoder, self).__init__()
        # self.ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1)  # 単語を単語ベクトルに変換する層
        self.ye = Embedding(vocab_size, embed_size, mask_zero=True)
        # self.eh = L.Linear(embed_size, 4 * hidden_size)  # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
        self.eh = Dense(4 * hidden_size, input_shape=(embed_size,))
        # self.hh = L.Linear(hidden_size, 4 * hidden_size)  # Decoderの中間ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
        self.hh = Dense(4 * hidden_size, input_shape=(hidden_size,))
        # self.fh = L.Linear(hidden_size, 4 * hidden_size)  # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
        self.fh = Dense(4 * hidden_size, input_shape=(hidden_size,))
        # self.bh = L.Linear(hidden_size, 4 * hidden_size)  # 順向きEncoderの中間ベクトルの加重平均を隠れ層の4倍のサイズのベクトルに変換する層
        self.bh = Dense(4 * hidden_size, input_shape=(hidden_size,))
        # self.he = L.Linear(hidden_size, embed_size)  # 隠れ層サイズのベクトルを単語ベクトルのサイズに変換する層
        self.he = Dense(embed_size, input_shape=(hidden_size,))
        # self.ey = L.Linear(embed_size, vocab_size)  # 単語ベクトルを語彙数サイズのベクトルに変換する層
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
            e = activations.tanh(self.ye(y))
            # 単語ベクトル、Decoderの中間ベクトル、順向きEncoderのAttention、逆向きEncoderのAttentionを使ってLSTM
            c, h = LSTM(c)(self.eh(e) + self.hh(h) + self.fh(f) + self.bh(b))
            # LSTMから出力された中間ベクトルを語彙数サイズのベクトルに変換する
            t = self.ey(activations.tanh(self.he(h)))
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
        sum_w = backend.zeros((self.batch_size, 1))  # ウェイトの合計値を計算するための値を初期化
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        for f, b in zip(fs, bs):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            w = activations.tanh(self.fh(f) + self.bh(b) + self.hh(h))
            w = backend.exp(self.hw(w))  # softmax関数を使って正規化する
            ws.append(w)  # 計算したウェイトを記録
            sum_w += w
        # 出力する加重平均ベクトルの初期化
        att_f = backend.zeros((self.batch_size, self.hidden_size))
        att_b = backend.zeros((self.batch_size, self.hidden_size))
        for f, b, w in zip(fs, bs, ws):
            w /= sum_w  # ウェイトの和が1になるように正規化
            # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
            att_f += Reshape((self.batch_size, self.hidden_size))(backend.batch_dot(f, w))
            att_b += Reshape((self.batch_size, self.hidden_size))(backend.batch_dot(b, w))
        return att_f, att_b


class AttSeq2Seq(object):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):
        """
        Sequence to Sequence with Attention Model
        :param vocab_size: 語彙数のサイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        """
        super(AttSeq2Seq, self).__init__()
        # self.inputs = Input(shape=(vocab_size,))

        self.f_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size)  # Forward Encoder
        self.b_encoder = LSTMEncoder(vocab_size, embed_size, hidden_size)  # Backward Encoder
        self.attention = Attention(hidden_size, batch_size)
        self.decoder = AttLSTMDecoder(vocab_size, embed_size, hidden_size)

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.fs = []  # 順向きのEncoderの中間ベクトル
        self.bs = []  # 逆向きのEncoderの中間ベクトル
        self.c = backend.zeros((batch_size, self.hidden_size))  # 内部メモリ
        self.h = backend.zeros((batch_size, self.hidden_size))  # 中間ベクトル

    def encode(self, words, batch_size):
        """
        Encoderの計算
        :param words: 入力で使用する単語記録されたリスト
        :param batch_size: ミニバッチのサイズ
        :return:
        """
        # 発話リスト内の単語をrowで分割して中身のみをnumpy arrayに変換
        words = [np.array(row, dtype='int32') for row in words]  # TODO: wordsがinput ???
        c = backend.zeros((batch_size, self.hidden_size))
        h = backend.zeros((batch_size, self.hidden_size))

        # 順向きのEncoderの計算
        for w in words:
            c, h = self.f_encoder(self.inputs(w), c, h)  # TODO: ここのw が，Inputになる！?
            self.fs.append(h)  # 計算された中間ベクトルを記録

        c = backend.zeros((batch_size, self.hidden_size))
        h = backend.zeros((batch_size, self.hidden_size))
        # 逆向きのEncoderの計算
        for w in reversed(words):
            c, h = self.b_encoder(w, c, h)  # TODO: ここのw が，Inputになる！?
            self.bs.insert(0, h)  # 計算された中間ベクトルを記録

        self.c = backend.zeros((batch_size, self.hidden_size))
        self.h = backend.zeros((batch_size, self.hidden_size))  # TODO: maybe h ???

    def decode(self, w):
        """
        Decoderの計算
        :param w: Decoderで入力する単語
        :return: 予測単語
        """
        att_f, att_b = self.attention(self.fs, self.bs, self.h)
        t, self.c, self.h = self.decoder(w, self.c, self.h, att_f, att_b)
        return t
