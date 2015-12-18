import dsd100
import argparse
import essentia.standard
import numpy as np


def compute_loudness(signal):
    audio = essentia.array(signal)
    loudness = essentia.standard.LoudnessVickers()
    w = essentia.standard.Windowing(type='hann')
    track_loudness = []
    for frame in essentia.standard.FrameGenerator(audio, frameSize=1024):
        track_loudness.append(loudness(w(frame)))

    return np.mean(track_loudness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse SISEC dataset')

    parser.add_argument(
        'dsd_folder',
        nargs='?',
        default=None,
        type=str,
        help='dsd 100 Folder'
    )

    args = parser.parse_args()

    dsd = dsd100.DB()

    for track in dsd.iter_dsd_tracks():
        print track.name
        print compute_loudness(track.audio.sum(axis=1))
