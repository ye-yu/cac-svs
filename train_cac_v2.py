from seq2seq_qi import SeqHParams, SequenceModel
from seq2seq_qi.utils import save_hparams
import os

SOS = "S"
EOS = "E"
PIT = "P"
DUR = "D"
RES = "R"


def abs_max(arr):
    if len(arr) < 1:
        return 0
    return max(arr)


if __name__ == '__main__':
    model_dir = "./trained-model/"
    model_increment = "{}".format(abs_max([int(i[-1]) for i in os.listdir(model_dir)]) + 1)
    model_destination_parent = os.path.join(model_dir, "model-{}".format(model_increment))
    os.mkdir(model_destination_parent)

    model_destination = os.path.join(model_destination_parent, "pitch-model")
    os.mkdir(model_destination)
    pitch_model_params = SeqHParams(
        # specifying the name of the model
        name="pitch-model",

        # specifying the source files
        train_file="dataset/training/pitch-time-text/pitch-input-file",
        train_vocab="dataset/training/pitch-time-text/vocabs/pitch-vocab",
        target_file="dataset/training/pitch-time-text/pitch-target-file",
        target_vocab="dataset/training/pitch-time-text/vocabs/pitch-vocab",

        # specifying flags
        sos=SOS,
        eos=EOS,

        # specifying encoder activations
        encoder_activation="sigmoid",
        decoder_activation="sigmoid",

        # specifying RNN cell type
        encoder_cell_type='lstm',
        decoder_cell_type='lstm',

        # specifying training parameters
        batch_size=128,
        embedding_size=512,
        rnn_units=1024,
        attention_units=1024,

        # logging destination
        log_destination=model_destination + "/log.csv"
    )

    save_hparams(pitch_model_params, model_destination + "/cac-model-hparams.json")

    pitch_model = SequenceModel(pitch_model_params)
    pitch_model.print_attr()
    pitch_model.train(18)
    pitch_model.save_model(model_destination)

    model_destination = os.path.join(model_destination_parent, "time-model")
    os.mkdir(model_destination)

    time_model_params = SeqHParams(
        # specifying the name of the model
        name="time-model",

        # specifying the source files
        train_file="dataset/training/pitch-time-text/pitch-input-file",
        train_vocab="dataset/training/pitch-time-text/vocabs/pitch-vocab",
        target_file="dataset/training/pitch-time-text/time-target-file",
        target_vocab="dataset/training/pitch-time-text/vocabs/time-vocab",

        # specifying flags
        sos=SOS,
        eos=EOS,

        # specifying encoder activations
        encoder_activation="sigmoid",
        decoder_activation="sigmoid",

        # specifying RNN cell type
        encoder_cell_type='lstm',
        decoder_cell_type='lstm',

        # specifying training parameters
        batch_size=128,
        embedding_size=512,
        rnn_units=1024,
        attention_units=1024,

        # logging destination
        log_destination=model_destination + "/log.csv"
    )

    save_hparams(time_model_params, model_destination + "/cac-model-hparams.json")

    time_model = SequenceModel(time_model_params)
    time_model.print_attr()
    time_model.train(18)
    time_model.save_model(model_destination)
