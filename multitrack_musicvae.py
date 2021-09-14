#Credits: Ian Simon, Adam Roberts, Colin Raffel, Jesse Engel, Curtis Hawthorne, Douglas Eck

# Setup
import numpy as np
import os
import tensorflow.compat.v1 as tf
from music21 import *
import magenta.music as mm
from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
tf.disable_v2_behavior()
print('Done!')

BATCH_SIZE = 1
Z_SIZE = 512
TOTAL_STEPS = 512
BAR_SECONDS = 2.0
CHORD_DEPTH = 49
SAMPLE_RATE = 44100
SF2_PATH = '~/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'
OUTPUT_PATH = ''
SOURCE_PATH = ''
temperature = 0.2   # min:0.01, max:1.5, step:0.1
num_bars = 32 # min:4, max:64, step:4

# Spherical linear interpolation
def slerp(p0, p1, t):
  omega = np.arccos(np.dot(np.squeeze(p0/np.linalg.norm(p0)), np.squeeze(p1/np.linalg.norm(p1))))
  so = np.sin(omega)
  return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

# Chord encoding tensor
def chord_encoding(chord):
  index = mm.TriadChordOneHotEncoding().encode_event(chord)
  c = np.zeros([TOTAL_STEPS, CHORD_DEPTH])
  c[0,0] = 1.0
  c[1:,index] = 1.0
  return c

# Trim sequences to exactly one bar
def trim_sequences(seqs, num_seconds=BAR_SECONDS):
  for i in range(len(seqs)):
    seqs[i] = mm.extract_subsequence(seqs[i], 0.0, num_seconds)
    seqs[i].total_time = num_seconds

# Consolidate instrument numbers by MIDI program
def fix_instruments_for_concatenation(note_sequences):
  instruments = {}
  for i in range(len(note_sequences)):
    for note in note_sequences[i].notes:
      if not note.is_drum:
        if note.program not in instruments:
          if len(instruments) >= 8:
            instruments[note.program] = len(instruments) + 2
          else:
            instruments[note.program] = len(instruments) + 1
        note.instrument = instruments[note.program]
      else:
        note.instrument = 9

# Chord-Conditioned Model
config = configs.CONFIG_MAP['hier-multiperf_vel_1bar_med_chords']
model = TrainedModel(
    config, batch_size=BATCH_SIZE,
    checkpoint_dir_or_path='~/model_chords_fb64.ckpt')

# Infer chords from musicxml file
musicxmlfile = mm.quantize_note_sequence(mm.musicxml_file_to_sequence_proto(SOURCE_PATH), 4)
chordlist = mm.infer_chords_for_sequence(musicxmlfile,
                              chords_per_bar=None,
                              key_change_prob=0.001,
                              chord_change_prob=0.5,
                              chord_pitch_out_of_key_prob=0.01,
                              chord_note_concentration=100.0,
                              add_key_signatures=False)

# Extract chords from musicxml file from command line input
xmlpath = str(input())
song = converter.parse(xmlpath).flatten()
songchords = song.getElementsByClass('Chord')
chordlenghts  = song.getElementsByClass('Durations')
chordlist = []
for i in range(0, len(songchords)):
  chordlist.append(str(songchords[i]).replace('<music21.harmony.ChordSymbol ', '').replace('>', ''))
  if i < len(songchords)-1:
    chordlist.append(songchords[i+1].offset-songchords[i].offset)
  else:
    chordlist.append(4.0)
chordlist = chordlist[:4]


# Same Style, Chord Progression

chords = chordlist

z = np.random.normal(size=[1, Z_SIZE])
seqs = [model.decode(length=TOTAL_STEPS, z=z, 
                     temperature=temperature,
                     c_input=chord_encoding(c))[0]
    for c in chords
]
trim_sequences(seqs)
fix_instruments_for_concatenation(seqs)
prog_ns = concatenate_sequences(seqs)
mm.plot_sequence(prog_ns)
mm.note_sequence_to_midi_file(prog_ns, OUTPUT_PATH + str(chords) + '.mid')


# Style Interpolation, Repeating Chord Progression

chords = chordlist

z1 = np.random.normal(size=[Z_SIZE])
z2 = np.random.normal(size=[Z_SIZE])
z = np.array([slerp(z1, z2, t)
              for t in np.linspace(0, 1, num_bars)])
seqs = [
    model.decode(length=TOTAL_STEPS, z=z[i:i+1, :],
                 temperature=temperature,
                 c_input=chord_encoding(chords[i % 4]))[0]
    for i in range(num_bars)
]
trim_sequences(seqs)
fix_instruments_for_concatenation(seqs)
prog_interp_ns = concatenate_sequences(seqs)
play(prog_interp_ns)
mm.plot_sequence(prog_interp_ns)

mm.note_sequence_to_midi_file(prog_ns, OUTPUT_PATH + str(chords) + 'Repeated' + '.mid')