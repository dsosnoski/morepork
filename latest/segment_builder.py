import sqlite3

import librosa
import numpy as np
import pickle

from training import training_parameters

frequency_min = 570
frequency_max = 1230
num_bands = int((frequency_max - frequency_min) / 10)
sample_span = 3.0

recordings_path = '/mnt/tmp/morepork/training-recordings'
samples_path = f'/mnt/tmp/morepork/training-samples-{num_bands}-{frequency_min}-{frequency_max}.pk'
db_path = '/home/dennis/projects/timserver/audio_analysis.db'
db_connection = None

try:
    db_connection = sqlite3.connect(db_path)
except sqlite3.Error as e:
    print(e)

try:
    sql = ''' SELECT DISTINCT recording_id FROM dennis_tags  '''
    cur = db_connection.cursor()
    cur.execute(sql)
    recordings = cur.fetchall()
except Exception as e:
    print(e, '\n')
    print('Failed to get tags\n')

positive_segments = {}
negative_segments = {}
for tuple in recordings:
    recording_id = tuple[0]
    frames, rate = librosa.load(f'{recordings_path}/{recording_id}.m4a', sr=None)
    # generate spectrogram
    nfft = int(rate / 10)
    stft = librosa.stft(frames, n_fft=nfft, hop_length=int(nfft / 2))
    npspec = np.abs(stft)[int(frequency_min / 10):int(frequency_max / 10)]

    try:
        sql = ''' SELECT start_time_seconds, finish_time_seconds, what FROM dennis_tags
                  WHERE dennis_tags.recording_id = ? AND dennis_tags.what <> 'maybepork'  '''
        cur = db_connection.cursor()
        cur.execute(sql, [int(recording_id)])
        tags = cur.fetchall()
    except Exception as e:
        print(e, '\n')
        print('Failed to get tags\n')

    for start, end, type in tags:
        limit = min(int((start + sample_span) * training_parameters.slices_per_second * 2), npspec.shape[1])
        base = max(int((end - sample_span) * training_parameters.slices_per_second * 2), 0)
        name = f'{recording_id}[{start:02.1f}-{end:02.1f}]'
        if limit - base >= training_parameters.slices_per_sample:
            segment = npspec[:, base:limit]
            if type == 'morepork':
                positive_segments[name] = segment
            else:
                negative_segments[name] = segment
        else:
            print(f'ignoring short sample {name}')

with open(samples_path, 'wb') as f:
    pickle.dump(positive_segments, f)
    pickle.dump(negative_segments, f)

print(f'built {len(positive_segments)} positive and {len(negative_segments)} negative samples')