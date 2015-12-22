from __future__ import division
import dsd100
import argparse
import essentia.standard
import numpy as np
from scipy.stats.mstats import gmean
import pandas as pd


def loundness_vickers(ndarray, win, hop):
    ess_array = essentia.array(ndarray)
    track_loudness = []
    loudness = essentia.standard.LoudnessVickers()
    w = essentia.standard.Windowing(type='hann')

    for frame in essentia.standard.FrameGenerator(
        ess_array,
        startFromZero=True,
        frameSize=win,
        hopSize=hop
    ):
        track_loudness.append(10.**(loudness(w(frame)) / 20.))

    return np.array(track_loudness)


def compute_mur_va_ranking(track, win=1024, overlap=0.5, exerpt_window=10.0):
    """Computes the masked loudness estimated over a sliding excerpt window
       for vocal accompaniment, only. Thus maximising the vocals transparency

    ml_vocal = max(loudness_mix - loudness_accompaniment, eps)

    References: https://www.spsc.tugraz.at/biblio/aichinger2011aichinger2011
    """
    if track.rate != 44100:
        raise ValueError("Samplerate not supported by Loudness algorithm")

    # compute framewise loudness
    audio_mix = track.audio.sum(axis=1)
    audio_acc = track.targets['accompaniment'].audio.sum(axis=1)

    track_loudness_mix = loundness_vickers(
        audio_mix, win, hop=int(win*overlap)
    )
    track_loudness_acc = loundness_vickers(
        audio_acc, win, hop=int(win*overlap)
    )

    ml_vocal = (track_loudness_mix - track_loudness_acc)

    ml_vocal = essentia.array(np.array(ml_vocal))

    exerpt_frame = int(np.ceil(
        exerpt_window * int(1 / overlap) / (win / float(track.rate))
    ))

    exerpt_hop = int(10)
    df = pd.DataFrame(columns=(
        'track_name',
        'loudness',
        'start_time',
        'stop_time'
        )
    )

    for i, frame in enumerate(essentia.standard.FrameGenerator(
        ml_vocal,
        startFromZero=True,
        frameSize=exerpt_frame,
        hopSize=exerpt_hop
    )):
        s = pd.Series(
            {
                'track_name': track.name,
                'cross_loudness': gmean(np.maximum(frame, np.finfo(float).eps)),
                'start_time': (i * exerpt_hop * win) / float(track.rate) / int(1 / overlap),
                'stop_time': ((i * exerpt_hop) + exerpt_frame) * win / float(track.rate) / int(1 / overlap),
            }
        )
        df = df.append(s, ignore_index=True)
    return df


def get_high_va_track(dsd):
    df = pd.DataFrame(columns=(
        'track_name',
        'cross_loudness',
        'start_time',
        'stop_time'
        )
    )

    for i, track in enumerate(dsd.iter_dsd_tracks()):
        print track.name
        df_l = compute_mur_va_ranking(track)
        df = df.append(df_l, ignore_index=True)
    return df


def get_top_n_exerpts(df, n=1):
    grouped = df.groupby(['track_name'])
    grouped.apply(lambda g: g.sort_index(by='cross_loudness', ascending=False).head(n))

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
    df = get_high_va_track(dsd)
    df.to_pickle("results.pandas")
    df_top1 = get_top_n_exerpts(df)
    df_top1.to_csv("top1.csv", sep='\t', encoding='utf-8')
