import gzip
import os

import librosa.display
import pickle

import numpy as np
import sqlite3
import tensorflow as tf

models = [
    '/hdd1/dennis/morepork-resnet34-rnn-3-4-6-7-7-2-2-max/weights3/model-928-0.9492',
    '/hdd1/dennis/morepork-resnet34-rnn-3-4-6-7-7-2-2-max/weights2/model-762-0.9482',
    '/hdd1/dennis/morepork-resnet34-epsilon001-7-7-2-2-max/weights1/model-896-0.9492'
]
pickle_file = '/hdd1/dennis/march-60-600-1200.pk'
results_path = '/hdd1/dennis/morepork/march-test-results'
test_limit = 0
accept_threshold = 0.5
slices_per_second = 20
seconds_per_sample = 3.0
slices_per_sample = int(slices_per_second * seconds_per_sample)
sample_slide_seconds = 1.0
sample_slide_slices = int(sample_slide_seconds * slices_per_second)

db_path_tags = '/hdd1/from_tim_for_dennis/audio_analysis_db4.db'
db_path_highlights = '/hdd1/dennis/audio_analysis.db'
db_connection_tags = None
db_connection_highlights = None
try:
    db_connection_tags = sqlite3.connect(db_path_tags)
    db_connection_highlights = sqlite3.connect(db_path_highlights)
except sqlite3.Error as e:
    print(e)

np.set_printoptions(linewidth=9999)
if not os.path.isdir(results_path):
    os.mkdir(results_path)

# initialize data used by analysis functions
with gzip.open(pickle_file, 'rb') as f:
    test_data = pickle.load(f)
print(f'loaded {len(test_data)} samples for tests')
error_counts = {}
error_in_class = {}
true_positives = {}
tag_cache = {}


def test_model(model_path, result_sums):
    model = tf.keras.models.load_model(model_path)
    for id, npspec in test_data.items():
        samples = []
        base_limit = npspec.shape[1] - slices_per_sample + sample_slide_slices
        for base in range(0, base_limit, sample_slide_slices):
            limit = base + slices_per_sample
            if limit > npspec.shape[1]:
                limit = npspec.shape[1]
            start = limit - slices_per_sample
            sample = npspec[:, start:limit]
            sample = librosa.amplitude_to_db(sample, ref=np.max)
            sample = sample / abs(sample.min()) + 1.0
            samples.append(sample.reshape(sample.shape + (1,)))
        samples = np.array(samples)
        predicts = model.predict(samples).flatten()
        if id in result_sums:
            result_sums[id] = result_sums[id] + predicts
        else:
            result_sums[id] = predicts
    tf.keras.backend.clear_session()


result_sums = {}
for model_path in models:
    test_model(model_path, result_sums)


def record_liklihood(id, liklihood, start, end):
    sql = ''' INSERT INTO liklihoods(recording_id, liklihood, start_second, end_second)
              VALUES(?, ?, ?, ?)  '''
    cur = db_connection_highlights.cursor()
    cur.execute(sql, (id, liklihood, start, end))

def find_likely_span(liklihoods, first, last):
    count = last - first
    if count == 0:
        end = first + seconds_per_sample
        return liklihoods[first], first, end
    elif count == 1:
        liklihood = max(liklihoods[first], liklihoods[last])
        end = first + seconds_per_sample
        return liklihood, first + 1, end
    elif count == 2:
        max_liklihood = max(liklihoods[first:last + 1])
        min_liklihood = min(liklihoods[first:last + 1])
        if max_liklihood == liklihoods[first + 1]:
            start = first + 1
            end = start + seconds_per_sample
            return max_liklihood, start, end
        elif min_liklihood == liklihoods[first]:
            start = first + 1
            end = last + seconds_per_sample
            return max_liklihood, start, end
        elif min_liklihood == liklihoods[last]:
            end = first + 1 + seconds_per_sample
            return max_liklihood, first, end
        else:
            end = last + seconds_per_sample
            return max_liklihood, first, end
    else:
        max_liklihood = max(liklihoods[first:last + 1])
        if max_liklihood > liklihoods[first]:
            start = first + 1
            if max_liklihood > liklihoods[last]:
                end = last - 1 + seconds_per_sample
                return max_liklihood, start, end
            else:
                end = last + seconds_per_sample
                return max_liklihood, start, end
        elif max_liklihood > liklihoods[last]:
            end = last - 1 + seconds_per_sample
            return max_liklihood, first, end
        else:
            start = first + 1
            end = last - 1 + seconds_per_sample
            return max_liklihood, start, end


for id, sum in result_sums.items():
    if any(x >= 1.0 for x in sum):
        first = -1
        liklihoods = [round(v * 33.33333333) for v in sum]
        for i in range(len(sum)):
            if sum[i] >= 1.0:
                if first < 0:
                    first = i
                last = i
            elif first >= 0:
                liklihood, start, end = find_likely_span(liklihoods, first, last)
                record_liklihood(id, liklihood, start, end)
                first = -1
        if first >= 0:
            liklihood, start, end = find_likely_span(liklihoods, first, last)
            record_liklihood(id, liklihood, start, end)

db_connection_highlights.commit()

