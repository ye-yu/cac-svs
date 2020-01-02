#! /usr/bin/python

import tensorflow as tf
import tensorflow_addons as tfa
import sys
import os
import numpy as np

__all__ = ["SequenceModel"]


class SeqHParams:
    def __init__(self,
                 train_file: str,
                 target_file: str,
                 train_vocab: str,
                 target_vocab: str,
                 sos: str,
                 eos: str,
                 **kwargs):
        self.name = kwargs.get("name", "seq2seq-model")
        self.train_file = train_file
        self.train_vocab = train_vocab
        self.target_file = target_file
        self.target_vocab = target_vocab
        self.sos = sos
        self.eos = eos

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

        self.encoder_activation = kwargs.get("encoder_activation", "sigmoid")
        self.decoder_activation = kwargs.get("decoder_activation", "sigmoid")

        self.embedding_size = kwargs.get("embedding_size", 128)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.attention_units = kwargs.get("attention_units", 128)

        self.optimizer = kwargs.get("optimizer", 'adam')
        self.learning_rate = kwargs.get("learning_rate", 0.001)

        self.batch_size = kwargs.get("batch_size", 64)
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

        self.train_dataset = [
            "{} {} {}".format(self.hparams.sos, i.strip(), self.hparams.eos) for i in self.train_dataset]
        self.target_dataset = [
            "{} {} {}".format(self.hparams.sos, i.strip(), self.hparams.eos) for i in self.target_dataset]

        self.train_tokenized_seq = self.train_tokenizer.texts_to_sequences(self.train_dataset)
        self.train_tokenized_seq = tf.keras.preprocessing.sequence.pad_sequences(self.train_tokenized_seq,
                                                                                 padding='post')
        self.train_vocab_size = len(self.train_tokenizer.word_index) + 1
        self.train_max_vocab_len = tf.math.reduce_max([len(t) for t in self.train_tokenized_seq])

        self.target_tokenized_seq = self.target_tokenizer.texts_to_sequences(self.target_dataset)
        self.target_tokenized_seq = tf.keras.preprocessing.sequence.pad_sequences(self.target_tokenized_seq,
                                                                                  padding='post')
        self.target_vocab_size = len(self.target_tokenizer.word_index) + 1
        self.target_max_vocab_len = tf.math.reduce_max([len(t) for t in self.target_tokenized_seq])

        self.batched_dataset = tf.data \
            .Dataset \
            .from_tensor_slices((self.train_tokenized_seq, self.target_tokenized_seq)) \
            .shuffle(len(self.train_dataset)) \
            .batch(self.hparams.batch_size, drop_remainder=True)

        self.encoder = Encoder(self.train_vocab_size,
                                self.hparams.embedding_size,
                                self.hparams.rnn_units,
                                self.hparams.encoder_cell,
                                self.hparams.encoder_dropout)

        self.decoder = Decoder(self.target_vocab_size,
                                self.hparams.embedding_size,
                                self.hparams.batch_size,
                                self.hparams.attention,
                                self.hparams.attention_units,
                                self.hparams.rnn_units,
                                self.hparams.decoder_cell)

        # self.encoder = _Encoder(self.train_vocab_size,
        #                         self.hparams.embedding_size,
        #                         self.hparams.rnn_units,
        #                         self.hparams.encoder_cell,
        #                         self.hparams.encoder_activation,
        #                         self.hparams.encoder_dropout)
        #
        # self.decoder = _Decoder(self.target_vocab_size,
        #                         self.hparams.embedding_size,
        #                         self.hparams.batch_size,
        #                         self.hparams.attention,
        #                         self.hparams.attention_normalized,
        #                         self.hparams.attention_units,
        #                         self.hparams.rnn_units,
        #                         self.hparams.decoder_cell,
        #                         self.hparams.decoder_activation,
        #                         self.hparams.decoder_dropout)
        #
        if self.hparams.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(lr=self.hparams.learning_rate)
        else:
            self.optimizer = self.hparams.optimizer(lr=self.hparams.learning_rate)

        self.steps = len(self.train_tokenized_seq) // self.hparams.batch_size
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def initialize_initial_state(self, batch_size):
        return [tf.zeros((batch_size, self.hparams.rnn_units)), tf.zeros((batch_size, self.hparams.rnn_units))]

    @tf.function
    def _train(self, input_batch, output_batch, encoder_initial_cell_state):
        with tf.GradientTape() as tape:
            encoder_emb_inp = self.encoder.encoder_embedding(input_batch)
            a, a_tx, c_tx = self.encoder.encoder_rnnlayer(encoder_emb_inp,
                                                          initial_state=encoder_initial_cell_state)

            decoder_input = output_batch[:, :-1]  # ignore <end>
            decoder_output = output_batch[:, 1:]  # ignore <start>

            decoder_emb_inp = self.decoder.decoder_embedding(decoder_input)

            self.decoder.attention_mechanism.setup_memory(a)
            decoder_initial_state = self.decoder.build_decoder_initial_state(self.hparams.batch_size,
                                                                             encoder_state=[a_tx, c_tx],
                                                                             dtype=tf.float32)

            outputs, _, _ = self.decoder.decoder(decoder_emb_inp,
                                                 initial_state=decoder_initial_state,
                                                 sequence_length=self.hparams.batch_size * [
                                                     self.target_max_vocab_len - 1]
                                                 )

            logits = outputs.rnn_output

            pred = tf.cast(tf.math.argmax(logits, axis=2), tf.int64)
            actu = tf.cast(decoder_output, tf.int64)
            accuracy = tf.math.count_nonzero(actu == pred) / (actu.shape[0] * actu.shape[1])

            loss = self.loss_function(decoder_output, logits)
            mask = tf.logical_not(tf.math.equal(decoder_output, 0))
            mask = tf.cast(mask, dtype=loss.dtype)
            loss = mask * loss
            loss = tf.reduce_mean(loss)

        variables = self.encoder.trainable_variables + self.encoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        grads_and_vars = zip(gradients, variables)
        self.optimizer.apply_gradients(grads_and_vars)
        return loss, accuracy

    def run_train(self, epochs, verbose=True):
        if verbose:
            def _print(*args, color=None, **kwargs):
                if color == 'red':
                    print('\033[31m', end='')
                elif color == 'green':
                    print('\033[32m', end='')
                print(*args, '\033[0m', **kwargs)

        else:
            def _print(*args, **kwargs):
                return

        best_acc = 0
        best_loss = np.inf
        _print("Training for {} epochs".format(epochs))
        total_batches = '?'
        for i in range(1, epochs + 1):
            encoder_state = self.initialize_initial_state(self.hparams.batch_size)
            total_loss = total_accuracy = 0.0
            batch = 0
            for (batch, (input_batch, output_batch)) in enumerate(self.batched_dataset.take(self.steps)):
                batch_loss, batch_accuracy = self._train(input_batch, output_batch, encoder_state)
                total_loss += batch_loss
                total_accuracy += batch_accuracy
                _print("\rEpoch {} Batch {}/{} [loss: {:0.04f}, accuracy: {:0.04f}]".format(i,
                                                                                            batch,
                                                                                            total_batches,
                                                                                            total_loss / batch,
                                                                                            total_accuracy / batch
                                                                                            ), end='')
            total_batches = batch + 1
            total_accuracy /= total_batches
            total_loss /= total_batches
            _print("\rEpoch {} [loss: {:0.04f}, accuracy: {:0.04f}]".format(i,
                                                                            total_loss,
                                                                            total_accuracy
                                                                            ), end='\n')
            _print("Best vs Last Accu: {:0.04f} -> {:0.04f}".format(best_acc, total_accuracy))
            _print("Best vs Last Loss: {:0.04f} -> {:0.04f}".format(best_loss, total_loss))

            if best_acc < total_accuracy:
                _print("Accuracy improved.", color='green', end='')
                best_acc = total_accuracy
            else:
                _print("Accuracy not improved.", color='red', end='')

            if best_loss > total_loss:
                _print("Loss improved.", color='green')
                best_loss = total_loss
            else:
                _print("Loss not improved", color='red')

            self.logger(i, total_loss, total_accuracy)

    @tf.function
    def infer_one(self, untokenized_sequence: list, beam_width: int):
        input_batch = tf.convert_to_tensor(self.target_tokenizer.texts_to_sequences([untokenized_sequence]))
        encoder_initial_cell_state = self.initialize_initial_state(1)
        encoder_emb_inp = self.encoder.encoder_embedding(input_batch)
        a, a_tx, c_tx = self.encoder.encoder_rnnlayer(encoder_emb_inp,
                                                      initial_state=encoder_initial_cell_state)

        decoder_input = tf.expand_dims([self.target_tokenizer.word_index[self.hparams.sos]] * 1, 1)
        self.decoder.decoder_embedding(decoder_input)

        # Build from attention
        encoder_memory = tfa.seq2seq.tile_batch(a, beam_width)
        self.decoder.attention_mechanism.setup_memory(encoder_memory)

        # Build decoder state from encoder last state
        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=1 * beam_width,
                                                                        dtype=tf.float32)
        encoder_state = tfa.seq2seq.tile_batch([a_tx, c_tx], multiplier=beam_width)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

        decoder_instance = tfa.seq2seq.BeamSearchDecoder(self.decoder.rnn_cell,
                                                         beam_width=beam_width,
                                                         output_layer=self.decoder.dense_layer)

        maximum_iterations = tf.round(tf.reduce_max(self.target_max_vocab_len) * tf.constant(2))

        decoder_embedding_matrix = self.decoder.decoder_embedding.variables[0]
        start_tokens = tf.fill([1], self.target_tokenizer.word_index[self.hparams.sos])
        end_token = self.target_tokenizer.word_index[self.hparams.eos]

        (first_finished, first_inputs, first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                                                                                  start_tokens=start_tokens,
                                                                                  end_token=end_token,
                                                                                  initial_state=decoder_initial_state)
        inputs = first_inputs
        state = first_state
        predictions = np.empty((1, beam_width, 0), dtype=np.int32)
        beam_scores = np.empty((1, beam_width, 0), dtype=np.float32)
        for j in range(maximum_iterations):
            beam_search_outputs, next_state, next_inputs, finished = decoder_instance.step(j, inputs, state)
            inputs = next_inputs
            state = next_state
            outputs = np.expand_dims(beam_search_outputs.predicted_ids, axis=-1)
            scores = np.expand_dims(beam_search_outputs.scores, axis=-1)
            predictions = np.append(predictions, outputs, axis=-1)
            beam_scores = np.append(beam_scores, scores, axis=-1)

        return self.target_tokenizer.sequences_to_texts(predictions[0]), tf.math.reduce_max(beam_scores[0], 1).numpy()

    def logger(self, epoch, loss, accuracy):
        if self.hparams.log_destination:
            with open(self.hparams.log_destination, 'a') as io:
                io.write("{},{},{}\n".format(epoch, loss, accuracy))

    def save_model(self, destination):
        try:
            os.mkdir(destination)
        except FileExistsError as e:
            pass
        destination = os.path.join(destination, self.hparams.name)
        self.encoder.save_weights(destination + ".encoder")
        self.decoder.save_weights(destination + ".decoder")

    def load_model(self, source):
        source = os.path.join(source, self.hparams.name)
        self.encoder.load_weights(source + ".encoder")
        self.decoder.load_weights(source + ".decoder")


class Encoder(tf.keras.Model):
    def __init__(self, input_vocab_size, embedding_dims, rnn_units, rnn_cell_type='lstm', dropout=0.0):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        if rnn_cell_type == 'lstm':
            self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,
                                                         return_sequences=True,
                                                         return_state=True,
                                                         dropout=dropout)
        elif rnn_cell_type == 'gru':
            self.encoder_rnnlayer = tf.keras.layers.GRU(rnn_units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        dropout=dropout)
        else:
            raise ValueError('Cell type unsupported:', rnn_cell_type)


class _Encoder(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 embedding_dims,
                 rnn_units,
                 rnn_cell_type,
                 activation,
                 dropout):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        if rnn_cell_type == 'lstm':
            self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,
                                                         return_sequences=True,
                                                         return_state=True,
                                                         dropout=dropout,
                                                         activation=activation)
        elif rnn_cell_type == 'gru':
            self.encoder_rnnlayer = tf.keras.layers.GRU(rnn_units,
                                                        return_sequences=True,
                                                        return_state=True,
                                                        dropout=dropout,
                                                        activation=activation)
        else:
            raise ValueError('Cell type unsupported:', rnn_cell_type)


class Decoder(tf.keras.Model):
    def __init__(self,
                 output_vocab_size,
                 embedding_dims,
                 batch_size,
                 attention,
                 dense_units,
                 rnn_units,
                 rnn_cell_type='lstm'):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims)

        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)

        if rnn_cell_type == 'lstm':
            self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        elif rnn_cell_type == 'gru':
            self.decoder_rnncell = tf.keras.layers.GRUCell(rnn_units)
        else:
            raise ValueError('Cell type unsupported:', rnn_cell_type)

        self.dense_units = dense_units
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(attention,
                                                                  None,
                                                                  batch_size * [output_vocab_size])
        self.rnn_cell = self.build_rnn_cell()
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell,
                                                sampler=self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, attention_type, memory, memory_sequence_length):
        if attention_type == 'luong':
            return tfa.seq2seq.LuongAttention(self.dense_units,
                                              memory=memory,
                                              memory_sequence_length=memory_sequence_length)
        elif attention_type == 'bahdanau':
            return tfa.seq2seq.BahdanauAttention(self.dense_units,
                                                 memory=memory,
                                                 memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell
    def build_rnn_cell(self):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell,
                                                self.attention_mechanism,
                                                attention_layer_size=self.dense_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


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
                 rnn_activation: str,
                 dropout: float):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims)

        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)

        if rnn_cell_type == 'lstm':
            self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units,
                                                            dropout=dropout,
                                                            activation=rnn_activation)
        elif rnn_cell_type == 'gru':
            self.decoder_rnncell = tf.keras.layers.GRUCell(rnn_units,
                                                           dropout=dropout,
                                                           activation=rnn_activation)
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
        self.rnn_cell = self.build_rnn_cell()
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

    def build_rnn_cell(self):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell,
                                                self.attention_mechanism,
                                                attention_layer_size=self.dense_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state


if sys.version_info.major != 3 and sys.version_info.minor != 6 and sys.version_info.micro != 9:
    raise Exception("Version mismatch: Use Python 3.6.9 and Tensorflow 2.0.0. Current interpreter version {}, "
                    "tensorflow version {}".format(sys.version, tf.__version__))
