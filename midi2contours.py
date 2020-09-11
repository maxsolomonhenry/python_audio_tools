# Quick, naive conversion of midi file into pitch/loudness contours for use as a contoller with DDSP. 
# Expects midi files with monophonic tracks.

# TODO: Couple note on/offs.

import mido
from mido import MidiFile
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

def midi2hz(notes):
    return 440.0 * (2.0**((notes - 69.0) / 12.0))

def midi2contours(midi_file):
    mid = MidiFile(midi_file)

    # Get requisite global info. Hopefully in first track.
    for msg in mid.tracks[0]:
        if msg.type == 'time_signature': 
            CLOCKS_PER_CLICK = msg.clocks_per_click
        if msg.type == 'set_tempo': 
            TEMPO = msg.tempo

    FRAMES_PER_S = 240
    DYNAMIC_RANGE_DB = 120.0
    SILENCE_DB = 100.0

    def ticks2secs(ticks):
        return mido.tick2second(ticks, CLOCKS_PER_CLICK, TEMPO)

    def ticks2frames(ticks):
        return int(ticks2secs(ticks)*FRAMES_PER_S)

    def velocity2db(velocity_track):
        return velocity_track/127. * DYNAMIC_RANGE_DB - SILENCE_DB

    def track2contours(midi_track):
        midi_pitch_track = np.array([])
        midi_velocity_track = np.array([])

        for msg in midi_track:
            if msg.time > 0 and msg.type[:4] == 'note':
                # Get length in frames.
                n_frames = ticks2frames(msg.time)

                # Repeat this note value for n_frames.
                midi_pitch_track = np.append(midi_pitch_track, np.repeat(msg.note, n_frames))

                # Also repeat velocity for n_frames.
                midi_velocity_track = np.append(midi_velocity_track, np.repeat(msg.velocity, n_frames))

        # Convert to Hz.
        f0_hz = midi2hz(midi_pitch_track)

        # Convert velocity to db.
        loudness_db = velocity2db(midi_velocity_track)

        return [f0_hz, loudness_db]

    contours = []

    print('Converting midi tracks...')
    for track in tqdm(mid.tracks):
        contours.append(track2contours(track))

    return contours

if __name__ == '__main__':
    # Test it out.
    contours = midi2contours('Untitled.mid')
