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
    seqs = list()
    for msg in midi.tracks[0]:
        msg.time = round(msg.time / 10)
        if 'note_on' in msg.type:
            seqs += [['R', msg.time]]
        elif 'note_off' in msg.type:
            seqs += [[msg.note, msg.time]]

    if seqs[0][0] == 'R':
        seqs = seqs[1:]

    new_seqs = list()

    for note, time in seqs:
        d, r = divmod(time, 100)
        if d:
            new_seqs += [[note, 100]] * d
        if r:
            new_seqs += [[note, r]]

    seqs = new_seqs

    for i, (note, time) in enumerate(seqs):
        assert time != 0

    return seqs


def make_training_dataset():
    # midi to csv
    # from dataset
    midi_dir = "dataset/midi"
    save_dest = "dataset/csv-4/training"
    for file in os.listdir(midi_dir):
        print(file)
        fpath = os.path.join(midi_dir, file)
        seqs = midi_to_sequence(fpath)
        with open(os.path.join(save_dest, file.replace('.', '_') + '.csv'), 'w') as io:
            for note, time in seqs:
                io.write("{},{}\n".format(note, time))


def make_inferring_dataset():
    midi_dir = "dataset/generating/midi/cleaned"
    save_dest = "dataset/csv-4/inferring"
    for file in os.listdir(midi_dir):
        print(file)
        fpath = os.path.join(midi_dir, file)
        seqs = midi_to_sequence(fpath)
        with open(os.path.join(save_dest, file.replace('.', '_') + '.csv'), 'w') as io:
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
    # make_training_dataset()
    # make_inferring_dataset()

    source_dir = 'dataset/csv-4/training'
    seq_length = 20

    dest = "dataset/training/v4"
    try:
        os.mkdir(dest)
    except FileExistsError as e:
        print("Directory already exists. Might overwrite content.")

    open(os.path.join(dest, "pitch-time-input-file"), 'w').close()
    open(os.path.join(dest, "pitch-time-target-file"), 'w').close()

    for file in os.listdir(source_dir):
        print(file)
        fpath = os.path.join(source_dir, file)
        with open(fpath) as io:
            csv = io.readlines()
        csv = [i.strip().split(',') for i in csv]

        for i, (n, d) in enumerate(csv[:-seq_length * 2]):
            if n == 'R':
                continue

            seg = csv[i:i + 40]

            starting_note = None
            prev_note = None
            seqs = list()
            for note, duration in seg:
                if note == 'R':
                    seqs += [[note, duration]]
                    continue
                note = int(note)
                if starting_note is None:
                    starting_note = note
                if prev_note is None:
                    note_diff = 0
                else:
                    note_diff = note - prev_note
                seqs += [[note_diff, duration]]
                prev_note = note

            assert seqs[0][0] == 0

            train, target = seqs[:20], seqs[20:]
            pitch_time_train = ' '.join([' '.join([str(a), str(b)]) for a, b in train])
            pitch_time_target = ' '.join([' '.join([str(a), str(b)]) for a, b in target][:10])

            dest = "dataset/training/v4"
            print(pitch_time_train, file=open(os.path.join(dest, "pitch-time-input-file"), 'a'))
            print(pitch_time_target, file=open(os.path.join(dest, "pitch-time-target-file"), 'a'))







