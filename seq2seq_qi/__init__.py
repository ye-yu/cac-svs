#! /usr/bin/python

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import os
import sys
import time

__all__ = ["SequenceModel"]

class SeqHParams:
    def __init__(self,
                  train_file: str,
                  target_file: str,
                  train_vocab: str,
                  target_vocab: str,
                  **kwargs):
        self.train_file = train_file
        self.train_vocab = train_vocab
        self.target_file = target_file
        self.target_vocab = target_vocab

        if kwargs.get("use_same_cell_type", False):
            self.encoder_cell = self.decoder_cell = kwargs.get("cell_type", "lstm")
        else:
            self.encoder_cell = kwargs.get("encoder_cell_type", "lstm")
            self.decoder_cell = kwargs.get("decoder_cell_type", "lstm")

        self.attention = kwargs.get("attention", "bahdanau")
        self.attention_normalized = kwargs.get("attention_normalized", False)

        if kwargs.get("use_same_cell_dropout", False):
            self.encoder_dropout = self.decoder_dropout = kwargs.get("dropout", 0.0)
        else:
            self.encoder_dropout = kwargs.get("encoder_dropout", 0.0)
            self.decoder_dropout = kwargs.get("decoder_dropout", 0.0)

        self.embedding_size = kwargs.get("embedding_size", 128)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.attention_units = kwargs.get("attention_units", 128)

        if kwargs.get("optimizer_learning_rate"):
            self.optimizer = kwargs.get("optimizer", tf.keras.optimizers.Adam(lr=kwargs.get("optimizer_learning_rate")))
        else:
            self.optimizer = kwargs.get("optimizer", tf.keras.optimizers.Adam())

        self.batch_size = kwargs.get("batch_size", 96)
        self.log_destination = kwargs.get("log_destination", None)


class SequenceModel:
    def __init__(self, hparams: SeqHParams):
        self.hparams = hparams
        self.train_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)
        self.target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)

        with open(self.hparams.train_vocab) as io:
            vocab = io.read().split()
        self.train_tokenizer.fit_on_texts(vocab)

        with open(self.hparams.target_vocab) as io:
            vocab = io.read().split()
        self.target_tokenizer.fit_on_texts(vocab)

        with open(self.hparams.train_file) as io:
            self.train_dataset = io.readlines()

        with open(self.hparams.target_file) as io:
            self.target_dataset = io.readlines()

        self.train_tokenized_seq = self.train_tokenizer.texts_to_sequences(self.train_dataset)
        self.train_tokenized_seq = tf.keras.preprocessing.sequence.pad_sequences(self.train_tokenized_seq,
                                                                                 padding='post')
        self.train_vocab_size = tf.math.reduce_max([len(t) for t in self.train_tokenized_seq])

        self.target_tokenized_seq = self.train_tokenizer.texts_to_sequences(self.target_dataset)
        self.target_tokenized_seq = tf.keras.preprocessing.sequence.pad_sequences(self.target_tokenized_seq,
                                                                                  padding='post')
        self.target_vocab_size = tf.math.reduce_max([len(t) for t in self.target_tokenized_seq])

        self.batched_dataset = (tf.data
                                  .Dataset
                                  .from_tensor_slices((self.train_tokenized_seq, self.target_tokenized_seq))
                                  .shuffle(len(self.train_dataset))
                                  .batch(self.hparams.batch_size, drop_remainder=True))

        self.encoder = _Encoder(self.train_vocab_size,
                                self.hparams.embedding_size,
                                self.hparams.rnn_units,
                                self.hparams.encoder_cell,
                                self.hparams.encoder_dropout)

        self.decoder = _Decoder(self.target_vocab_size,
                                self.hparams.embedding_size,
                                self.hparams.batch_size,
                                self.hparams.attention,
                                self.hparams.attention_normalized,
                                self.hparams.attention_units,
                                self.hparams.rnn_units,
                                self.hparams.decoder_cell,
                                self.hparams.encoder_dropout)

        self.buffer_size = self.hparams.batch_size // len(self.train_tokenized_seq)

    @tf.function
    def train(self):
        pass

    @tf.function
    def infer_one(self):
        pass

    def save_model(self, destination):
        pass

    def load_model(self, source):
        pass


class _Encoder(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 embedding_dims,
                 rnn_units,
                 rnn_cell_type,
                 dropout):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        if rnn_cell_type == 'lstm':
            self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,
                                                         return_sequences=True,
                                                         return_state=True,
                                                         dropout=dropout)
        elif rnn_cell_type == 'gru':
            self.encoder_rnnlayer = tf.keras.layers.gru(rnn_units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        dropout=dropout)
        else:
            raise ValueError('Cell type unsupported:', rnn_cell_type)


class _Decoder(tf.keras.Model):
    def __init__(self,
                 output_vocab_size: int,
                 embedding_dims: int,
                 batch_size: int,
                 attention: str,
                 attention_norm: bool,
                 dense_units: int,
                 rnn_units: int,
                 rnn_cell_type: str,
                 dropout: float):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims)

        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)

        if rnn_cell_type == 'lstm':
            self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units, dropout=dropout)
        elif rnn_cell_type == 'gru':
            self.decoder_rnncell = tf.keras.layers.GRUCell(rnn_units, dropout=dropout)
        else:
            raise ValueError('Cell type unsupported:', rnn_cell_type)

        self.dense_units = dense_units
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(attention,
                                                                  attention_norm,
                                                                  None,
                                                                  batch_size * [output_vocab_size])
        self.rnn_cell = self.build_rnn_cell(batch_size)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell,
                                                sampler=self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, attention_type, norm, memory, memory_sequence_length):
        if attention_type == 'luong':
            return tfa.seq2seq.LuongAttention(self.dense_units,
                                              memory=memory,
                                              memory_sequence_length=memory_sequence_length,
                                              normalize=norm)
        elif attention_type == 'bahdanau':
            return tfa.seq2seq.BahdanauAttention(self.dense_units,
                                                 memory=memory,
                                                 memory_sequence_length=memory_sequence_length,
                                                 normalize=norm)

    # wrap decodernn cell
    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell,
                                                self.attention_mechanism,
                                                attention_layer_size=self.dense_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


if sys.version_info.major != 3 and sys.version_info.minor != 6 and sys.version_info.micro != 9:
    raise Exception("Version mismatch: Use Python 3.6.9 and Tensorflow 2.0.0. Current interpreter version {}, "
                    "tensorflow version {}".format(sys.version, tf.__version__))

