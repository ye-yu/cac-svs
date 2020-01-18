import numpy as np
import os
import mido

SOS = "S"
EOS = "E"
PIT = "P"
DUR = "D"
RES = "R"


def midi_to_sequence(source):
    midi = mido.MidiFile(source)
    starting_note = None
    prev_note = None
    seqs = list()
    for msg in midi.tracks[0]:
        msg.time = round(msg.time / 10)
        if 'note_on' in msg.type:
            seqs += [['R', msg.time]]
        elif 'note_off' in msg.type:
            if starting_note is None:
                starting_note = msg.note
            if prev_note is None:
                note_diff = 0
            else:
                note_diff = msg.note - prev_note
            seqs += [[note_diff, msg.time]]
            prev_note = msg.note

    if seqs[0][0] == 'R':
        seqs = seqs[1:]

    new_seqs = list()

    for note, time in seqs:
        d, r = divmod(time, 100)
        for _ in range(d):
            new_seqs += [[note, 100]]
            note = 0
        if r:
            new_seqs += [[note, r]]

    seqs = new_seqs

    for note, time in seqs:
        assert time != 0

    return starting_note, seqs


def make_dataset():
    # midi to csv
    # from dataset
    midi_dir = "dataset/midi"
    save_dest = "dataset/csv-3"
    for file in os.listdir(midi_dir):
        print(file)
        fpath = os.path.join(midi_dir, file)
        snote, seqs = midi_to_sequence(fpath)
        with open(os.path.join(save_dest, file.replace('.', '_') + '.csv'), 'w') as io:
            for note, time in seqs:
                io.write("{},{}\n".format(note, time))


def make_generating_input():
    midi_dir = "dataset/generating/midi/cleaned"
    save_dest = "dataset/generating/csv"
    for file in os.listdir(midi_dir):
        print(file)
        fpath = os.path.join(midi_dir, file)
        snote, seqs = midi_to_sequence(fpath)
        with open(os.path.join(save_dest, file.replace('.', '_') + "startwith{}".format(snote) +'.csv'), 'w') as io:
            for note, time in seqs:
                io.write("{},{}\n".format(note, time))


def segment(arr, length):
    new_arr = list()
    d, r = divmod(len(arr), length)
    print("segmenting for:", d, r)
    for i in range(d):
        new_arr += [arr[i*length:(i+1)*length]]
    if r:
        new_arr += [arr[-r:]]
    return new_arr


def wrap_seq(seq, for_target=False):
    if for_target:
        return "{} {} {}".format(SOS, ' '.join("{} {} {} {}".format(PIT, note, DUR, time) for note, time in seq), EOS)
    return ' '.join("{} {} {} {}".format(PIT, note, DUR, time) for note, time in seq)


if __name__ == '__main__':
    source_dir = 'dataset/csv-3'
    pitch_vocab = set()
    time_vocab = set()
    seq_length = 20

    dest = "dataset/training/pitch-time-text"
    open(os.path.join(dest, "input-file"), 'w').close()
    open(os.path.join(dest, "target-file"), 'w').close()

    for file in os.listdir(source_dir):
        print(file)
        fpath = os.path.join(source_dir, file)
        with open(fpath) as io:
            csv = io.readlines()
        csv = [i.strip().split(',') for i in csv]
        pitch_vocab = pitch_vocab.union([i[0] for i in csv])
        time_vocab = time_vocab.union([i[1] for i in csv])

        for skips in range(seq_length):
            csv = csv[skips:]
            segments = segment(csv, seq_length)

            input_seg = segments[:-1]
            target_seg = segments[1:]

            for i in range(len(input_seg)):
                while input_seg[i][0][0] == 'R':
                    input_seg[i] = input_seg[i][1:]

            for train, target in zip(input_seg, target_seg):
                if len(train) < 1:
                    continue
                with open(os.path.join(dest, "input-file"),
                          'a') as train_io, open(os.path.join(dest, "target-file"),
                                                 'a') as target_io:
                    print(wrap_seq(train, False), file=train_io)
                    print(wrap_seq(target, True), file=target_io)

    pitch_vocab = list(pitch_vocab - {RES})
    pitch_vocab = np.array(pitch_vocab).astype(int)
    p_max = np.max(np.abs(pitch_vocab))

    full_vocab = ' '.join([SOS, EOS, PIT, DUR, RES] + list((np.arange(-p_max, 101)).astype(str)))
    with open("dataset/generating/text/full-vocab", 'w') as io:
        print(full_vocab, file=io)
