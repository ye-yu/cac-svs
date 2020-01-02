import json
from seq2seq_qi import SeqHParams


def generate_uniques(*files):
    unique = set()
    for file in files:
        with open(file) as fio:
            lines = fio.readlines()
            lines = [line.strip().split() for line in lines]
            for line in lines:
                unique = unique.union(set(line))
    return unique


def save_hparams(hparams: SeqHParams, destination):
    json.dump(hparams.__dict__, open(destination, 'w'), indent=4)


def load_hparams(source):
    params = json.load(open(source, 'r'))
    return SeqHParams(**params)
