import seq2seq_qi
import seq2seq_qi.utils
import numpy as np
from collections import namedtuple

# start of sequence
SOS = 'S'

# end of sequence
EOS = 'E'

# rest note
RES = 'R'


def make_pitch_vocab(destination, *sources):
    unique = seq2seq_qi.utils.generate_uniques(*sources)

    inp_int = np.array(list(unique - {RES})).astype(int)
    inp_max = np.max(np.abs(inp_int))

    vocab_range = list((np.arange(inp_max * 2 + 1) - inp_max).astype(str))
    vocab_range = vocab_range + [RES, SOS, EOS]

    with open(destination, 'w') as io:
        io.write(' '.join(vocab_range))


def make_time_vocab(destination, *sources):
    unique = seq2seq_qi.utils.generate_uniques(*sources)

    inp_int = np.array(list(unique - {RES})).astype(int)
    inp_max = np.max(np.abs(inp_int))

    vocab_range = list((np.arange(inp_max + 1) - inp_max).astype(str))
    vocab_range = vocab_range + [RES, SOS, EOS]

    with open(destination, 'w') as io:
        io.write(' '.join(vocab_range))


if __name__ == "__main__":
    # make_pitch_vocab("dataset/text/vocabs/pitch-vocab",
    #                  "dataset/text/pitch-input",
    #                  "dataset/text/pitch-target")
    #
    # make_time_vocab("dataset/text/vocabs/time-vocab",
    #                 "dataset/text/time-target")

    model_destination = "dataset/trained-model/pitch-model/model-5"
    p_model_params = seq2seq_qi.SeqHParams(
        # specifying the name of the model
        name="pitch-model",

        # specifying the source files
        train_file="dataset/text/pitch-input",
        train_vocab="dataset/text/vocabs/pitch-vocab",
        target_file="dataset/text/pitch-target",
        target_vocab="dataset/text/vocabs/pitch-vocab",

        # specifying flags
        sos=SOS,
        eos=EOS,

        # specifying encoder activations
        encoder_activation="sigmoid",
        decoder_activation="sigmoid",

        # specifying training parameters
        batch_size=64,
        embedding_size=128,
        rnn_units=128,
        attention_units=128,

        # logging destination
        log_destination=model_destination + "/log.csv"
    )

    seq2seq_qi.utils.save_hparams(p_model_params, model_destination + "/p-model-hparams.json")

    p_model = seq2seq_qi.SequenceModel(p_model_params)
    p_model.train(15)
    p_model.save_model(model_destination)
