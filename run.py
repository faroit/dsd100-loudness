from __future__ import division
import dsd100
import argparse
import essentia.standard
import numpy as np
import scipy.stats.mstats
import pandas as pd


def compute_mur_va_ranking(track, framesize=1024, exerpt_window=10.0):
    """Computes the masked to unmasked loudness ratio over a sliding excerpt window
       for vocal accompaniment, only. Thus maximising the vocals transparency

                        max(loudness_mix - loudness_accompaniment, 0.003)
      MUR_vocal = -------------------------------------------------------------
                                    max(loudness_voc, 0.003)

    References: https://www.spsc.tugraz.at/biblio/aichinger2011aichinger2011
    """
    if track.rate != 44100:
        raise ValueError("Samplerate not supported by Loudness algorithm")

    # compute framewise loudness
    audio_mix = essentia.array(
        track.audio.sum(axis=1)
    )
    audio_voc = essentia.array(
        track.targets['vocals'].audio.sum(axis=1)
    )
    audio_acc = essentia.array(
        track.targets['accompaniment'].audio.sum(axis=1)
    )

    loudness = essentia.standard.LoudnessVickers()
    w = essentia.standard.Windowing(type='hann')
    track_loudness_mix = []
    track_loudness_voc = []
    track_loudness_acc = []

    for frame in essentia.standard.FrameGenerator(
        audio_mix,
        startFromZero=True,
        frameSize=framesize,
        hopSize=512
    ):
        track_loudness_mix.append(10.**(loudness(w(frame)) / 20.))

    for frame in essentia.standard.FrameGenerator(
        audio_voc,
        startFromZero=True,
        frameSize=framesize,
        hopSize=512
    ):
        track_loudness_voc.append(10.**(loudness(w(frame)) / 20.))

    for frame in essentia.standard.FrameGenerator(
        audio_acc,
        startFromZero=True,
        frameSize=framesize,
        hopSize=512
    ):
        track_loudness_acc.append(10.**(loudness(w(frame)) / 20.))

    track_loudness_lin_mix = np.array(track_loudness_mix)
    track_loudness_lin_acc = np.array(track_loudness_acc)
    track_loudness_lin_voc = np.array(track_loudness_voc)

    track_mur = (track_loudness_lin_mix - track_loudness_lin_acc) \
        / track_loudness_lin_voc

    track_mur = essentia.array(np.array(track_mur))
    exerpt_frame = int(np.ceil(10.0 * 2 / (framesize / float(track.rate))))

    exerpt_hop = 10
    df = pd.DataFrame(columns=(
        'track_name',
        'loudness',
        'start_time',
        'stop_time'
        )
    )

    for i, frame in enumerate(essentia.standard.FrameGenerator(
        track_mur,
        startFromZero=True,
        frameSize=exerpt_frame,
        hopSize=exerpt_hop
    )):
        s = pd.Series(
            {
                'track_name': track.name,
                'loudness': scipy.stats.mstats.gmean(np.maximum(frame, np.finfo(float).eps)),
                'start_time': (i * exerpt_hop) * framesize / float(track.rate) / 2,
                'stop_time': ((i * exerpt_hop) + exerpt_frame) * framesize / float(track.rate) / 2,
            }
        )
        df = df.append(s, ignore_index=True)
    return df


def get_high_va_track(dsd):
    df = pd.DataFrame(columns=(
        'track_name',
        'loudness',
        'start_time',
        'stop_time'
        )
    )

    for i, track in enumerate(dsd.iter_dsd_tracks()):
        print track.name
        df_l = compute_mur_va_ranking(track)
        print df_l.ix[df_l['loudness'].idxmax()]
        df.append(df_l, ignore_index=True)
    return df


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
    get_high_va_track(dsd)
