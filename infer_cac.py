import pandas as pd
from aaapi import MidiUtils
import os

SOS = "S"
EOS = "E"
PIT = "P"
DUR = "D"
RES = "R"

if __name__ == '__main__':
    source = 'trained-model/seq2seq-0.4-5/'

    csv_source = source + 'csv'
    midi_dest = source + 'midi'
    # for i in os.listdir(csv_source):
    #     fpath = os.path.join(csv_source, i)
    #     csv = pd.read_csv(fpath)
    #     csv = csv[['note', 'duration']]
    #     start_with = int(i[:-4].split('startwith')[1])
    #
    #     notes = (csv[csv['note'] != 'R']['note'].astype(int))
    #     csv.loc[csv['note'] == 'R', 'note'] = -1
    #     notes[0] = start_with
    #     notes = notes.cumsum()
    #     csv.loc[notes.index, 'note'] = notes
    #
    #     csv['duration'] = csv['duration'] * 10
    #     csv['note'] = csv['note'].astype(int)
    #     MidiUtils.csv_to_midi(csv, os.path.join(midi_dest, i.replace('.', '_')))

    for i in os.listdir(csv_source):
        fpath = os.path.join(csv_source, i)
        csv = pd.read_csv(fpath)
        csv = csv[['note', 'duration']]

        csv.loc[csv['note'] == 'R', 'note'] = -1

        csv['duration'] = csv['duration'] * 10
        csv['note'] = csv['note'].astype(int)

        csv = csv[(csv['note'] <= 127)]
        csv = csv[(csv['duration'] >= 0)]
        try:
            MidiUtils.csv_to_midi(csv, os.path.join(midi_dest, i.replace('.', '_')))
        except ValueError as ve:
            print("Cannot parse and convert", i)
            print(csv)