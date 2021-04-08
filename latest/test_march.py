import gzip
import os

import librosa.display
import pickle
import pylab

import numpy as np
import sqlite3
import tensorflow as tf
#root_dir = '/hdd1/dennis/morepork-resnet34-rnn-3-4-7-7-2-2-max'
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


def sample_key(id, start):
    return f'{id}[{start:02.02f}-{start+seconds_per_sample:02.02f}]'

def count_error(id, start, in_class, level):
    key = sample_key(id, start)
    if key in error_counts:
        list = error_counts[key]
        list.append(level)
        if error_in_class[key] != in_class:
            print(f'ERROR: {key} has both positive and negative errors')
    else:
        error_counts[key] = [level]
        error_in_class[key] = in_class

def count_true_positive(id, start, level):
    key = sample_key(id, start)
    if key in true_positives:
        list = true_positives[key]
        list.append(level)
    else:
        true_positives[key] = [level]

def save_spectfig(spec, name, title):
    librosa.display.specshow(np.squeeze(spec))
    pylab.title(title)
    pylab.savefig(f'/tmp/{name}.png', bbox_inches=None, pad_inches=0)
    pylab.close()

def get_tags(recording_id):
    if recording_id in tag_cache:
        return tag_cache[recording_id]
    else:
        try:
            sql = ''' SELECT start_time_seconds, finish_time_seconds, what FROM training_validation_test_data
                      WHERE training_validation_test_data.recording_id = ? AND (training_validation_test_data.what LIKE 'morepork%' OR training_validation_test_data.what LIKE 'maybe_morepork%')
					  ORDER BY training_validation_test_data.start_time_seconds  '''
            cur = db_connection_tags.cursor()
            cur.execute(sql, [int(recording_id)])
            tags = cur.fetchall()
            tag_cache[recording_id] = tags
            return tags
        except Exception as e:
            print(e, '\n')
            print('Failed to get tags ' + str(recording_id), '\n')

def test_weights(model_path, out_file):
    model = tf.keras.models.load_model(model_path)
    print(f'Test results for {model_path}:')
    out_file.write(f'\nTest results for {model_path}:\n')
    test_count = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    match_counts = np.zeros([2,2,2], dtype=int)
    for id, npspec in test_data.items():
        samples = []
        last_sample_time = None
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
            last_sample_time = start / slices_per_second
        samples = np.array(samples)
        values = model.predict(samples).flatten()
        predicts = np.array([p >= accept_threshold for p in values], dtype=int)
        possibles = np.zeros(len(samples), dtype=int)
        confirmed = np.zeros(len(samples), dtype=int)
        last_sample_index = len(samples) - 1

        def set_sample_flags(start, end, strict, flags):
            # first sample including any part of this time span
            min_index = max(0, min(int((start - seconds_per_sample) / sample_slide_seconds + .999), last_sample_index))
            if start >= last_sample_time:
                min_index = min(min_index, last_sample_index)
            # last sample including any part of this time span
            max_index = min(int(end / sample_slide_seconds), last_sample_index)
            if end >= last_sample_time:
                max_index = last_sample_index
            count = 0
            for i in range(min_index, max_index + 1):
                base = i * sample_slide_seconds
                if not strict or (base <= start and (base + seconds_per_sample) >= end):
                    flags[i] = 1
                    count += 1
            return count

        tags = get_tags(id)
        for i in range(len(tags)):
            start = tags[i][0]
            end = tags[i][1]
            what = tags[i][2]
            test_count -= set_sample_flags(start, end, False, possibles)
            if what == 'morepork_more-pork':
                test_count += set_sample_flags(start, end, True, confirmed)

        test_count += len(samples)
        has_error = False
        errors = np.zeros(len(samples), dtype=int)
        for i in range(len(samples)):
            match_counts[predicts[i], possibles[i], confirmed[i]] += 1
            start_time = i * sample_slide_seconds
            if start_time > last_sample_time:
                start_time = last_sample_time
            if predicts[i] and not possibles[i]:
                false_positive += 1
                # save_spectfig(samples[i], f'{id}-{start_time:02.2f}-{start_time+seconds_per_sample:02.2f}',
                #               f'false positive {values[i]}')
                count_error(id, start_time, False, values[i])
                has_error = True
                errors[i] = 1
            elif confirmed[i]:
                if predicts[i]:
                    true_positive += 1
                else:
                    false_negative += 1
                    # save_spectfig(samples[i], f'{id}-{start_time:02.2f}-{start_time + seconds_per_sample:02.2f}',
                    #               f'false negative {values[i]}')
                    count_error(id, start_time, True, values[i])
                    has_error = True
                    errors[i] = 2
        if has_error:
            column_high = np.zeros(len(samples), dtype=int)
            column_low = np.zeros(len(samples), dtype=int)
            for i in range(len(samples)):
                column_high[i] = int(i/10)
                column_low[i] = i % 10
            out_file.write(f' error(s) in predictions for {id}:\n')
            text = str(column_high).replace('[', ' ').replace(']', ' ')
            out_file.write(f'              {text}\n')
            text = str(column_low).replace('[', ' ').replace(']', ' ')
            out_file.write(f'              {text}\n')
            out_file.write(f'  Predicted:  {predicts}\n')
            out_file.write(f'  Possibles:  {possibles}\n')
            out_file.write(f'  Confirmed:  {confirmed}\n')
            markers = str(errors).replace('0', ' ').replace('1', '+').replace('2', '-')
            out_file.write(f'              {markers}\n')
    num_wrong = false_negative + false_positive
    print(f' total of {test_count - num_wrong} matching and {num_wrong} non-matching predictions')
    out_file.write(f' total of {test_count - num_wrong} matching and {num_wrong} non-matching predictions\n')
    out_file.write(' Predict  Possible Confirm')
    for i0 in range(2):
        for i1 in range(2):
            for i2 in range(2):
                    #print(f' {str(bool(i0)).ljust(8)} {str(bool(i1)).ljust(8)} {str(bool(i2)).ljust(8)} = {match_counts[i0, i1, i2]}')
                    out_file.write(f' {str(bool(i0)).ljust(8)} {str(bool(i1)).ljust(8)} {str(bool(i2)).ljust(8)} = {match_counts[i0, i1, i2]}\n')
    if true_positive != 0:
        print(f' true positives {true_positive}, false positives {false_positive}, false negatives {false_negative}, out of total {test_count}')
        out_file.write(f' true positives {true_positive}, false positives {false_positive}, false negatives {false_negative}, out of total {test_count}\n')
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fscore = 2. * precision * recall / (precision + recall)
        print(f' precision {precision:.4f}, recall {recall:.4f}, f-score {fscore:.4f}')
        out_file.write(f' precision {precision:.4f}, recall {recall:.4f}, f-score {fscore:.4f}\n')
    else:
        print(f' no correct predictions')
        out_file.write(f' no correct predictions\n')
    #print(f' {false_positive} possible morepork found, total distinct possibles {len(error_counts)}')
    out_file.write(f' {false_positive} possible morepork found, total distinct possibles {len(error_counts)}')
    return precision, recall, fscore

def listPaths(basepath):
    namelist = os.listdir(basepath)
    pathlist = list()
    for name in namelist:
        namepath = os.path.join(basepath, name)
        if os.path.isdir(namepath):
            pathlist = pathlist + listPaths(namepath)
        else:
            pathlist.append(namepath)
    return sorted(pathlist)


def test_model(root_path):
    _, model_dir = os.path.split(root_path)
    results_file = f'{results_path}/test-{model_dir}.txt'
    fscore_max = 0.0
    weights_count = 0
    results = []
    with open(results_file, 'w') as f:
        print(f'\nTesting model {root_path}')
        f.write(f'\nTesting model {root_path}\n')
        model_paths = [os.path.dirname(p) for p in listPaths(root_path) if p.endswith('saved_model.pb')]
        for model_path in model_paths:
            weights_count += 1
            results.append(test_weights(model_path, f))
        results = np.array(results)
        means = np.mean(results, axis=0)
        stds = np.std(results, axis=0)
        maxs = np.max(results, axis=0)
        print(f'\nMean precision {means[0]:.4f}, recall {means[1]:.4f}, f-score {means[2]:.4f}')
        f.write(f'\nMean precision {means[0]:.4f}, recall {means[1]:.4f}, f-score {means[2]:.4f}')
        print(f'\nStandard deviation precision {stds[0]:.4f}, recall {stds[1]:.4f}, f-score {stds[2]:.4f}')
        f.write(f'\nStandard deviation precision {stds[0]:.4f}, recall {stds[1]:.4f}, f-score {stds[2]:.4f}')
        print(f'\nMaximum precision {maxs[0]:.4f}, recall {maxs[1]:.4f}, f-score {maxs[2]:.4f}')
        f.write(f'\nMaximum precision {maxs[0]:.4f}, recall {maxs[1]:.4f}, f-score {maxs[2]:.4f}')

    tf.keras.backend.clear_session()
    return weights_count


test_model('/hdd1/dennis/morepork-resnet34-7-7-2-2-max-randomtestval1')
test_model('/hdd1/dennis/morepork-resnet34-7-7-2-2-max-fixedtestval1')

# with open(errors_path, 'w') as f:
#     json.dump(error_counts, f)
# sorted_errors = sorted(error_counts.items(), key = lambda x: len(x[1]), reverse = True)
# cutoff = 1
# top_errors = [x for x in sorted_errors if len(x[1]) >= cutoff]
# print('\n\nMost common errors:')
# for error in top_errors:
#     if error_in_class[error[0]]:
#         truth = 'negative'
#     else:
#         truth = 'positive'
#     print(f' classified {error[0]} as false {truth} {len(error[1])} times: {error[1]}')
# for error in sorted_errors:
#     (id, span) = error[0].split('[')
#     start = float(span.split('-')[0])
#     if error_in_class[error[0]]:
#         predict = 'notpork'
#     else:
#         predict = 'morepork'
#     sql = ''' INSERT INTO dennis_march(recording_id, slice_start_time, number_of_times, prediction, checked)
#               VALUES(?, ?, ?, ?, ?)  '''
#     cur = db_connection_highlights.cursor()
#     cur.execute(sql, (int(id), start, len(error[1]), predict, 0))
# db_connection_highlights.commit()

