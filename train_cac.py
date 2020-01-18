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
    model_destination = os.path.join(model_dir, "model-{}".format(model_increment))
    os.mkdir(model_destination)
    cac_model_params = SeqHParams(
        # specifying the name of the model
        name="cac-model",

        # specifying the source files
        train_file="dataset/generating/text/input-file",
        train_vocab="dataset/generating/text/full-vocab",
        target_file="dataset/generating/text/target-file",
        target_vocab="dataset/generating/text/full-vocab",

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
        batch_size=512,
        embedding_size=512,
        rnn_units=512,
        attention_units=512,

        # logging destination
        log_destination=model_destination + "/log.csv"
    )

    save_hparams(cac_model_params, model_destination + "/cac-model-hparams.json")

    cac_model = SequenceModel(cac_model_params)
    cac_model.print_attr()
    cac_model.train(18)
    cac_model.save_model(model_destination)
