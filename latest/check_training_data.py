
import numpy as np
import random
import sqlite3
import tensorflow as tf

import morepork_support
import training_parameters

trainfract = .5
resnet_size = 34
conv_size = (5,5)
conv_strides = (2,2)
max_pooling = True
max_noise = None
if max_pooling:
    pooling = 'max'
else:
    pooling = 'avg'
db_path = '/hdd1/dennis/audio_analysis.db'
db_connection = None

try:
    db_connection = sqlite3.connect(db_path)
except sqlite3.Error as e:
    print(e)


def check_predicts(results, actuals, names, errors):
    predicts = np.array([p >= 0.5 for p in results], dtype=int)
    for i in range(len(predicts)):
        if predicts[i] != actuals[i]:
            name = names[i]
            if not name in errors:
                errors[name] = 0
            errors[name] += 1

recording_samples = morepork_support.load_recording_samples()
recording_ids = list(recording_samples.keys())
random_generator = np.random.default_rng()
errors = {}
for i in range(20):
    # set up samples to be used for training
    random.shuffle(recording_ids)
    count = int(len(recording_ids) / 2)
    count1, ds1 = morepork_support.build_dataset(recording_ids[:count], recording_samples)
    model = morepork_support.build_model(resnet_size, conv_size, conv_strides, max_pooling)
    training_steps = count1 // training_parameters.batch_size
    model.fit(ds1, steps_per_epoch=training_steps, epochs=1000, verbose=0)
    x, y, names = morepork_support.build_test(recording_ids[count:], recording_samples)
    results = model.predict(x).flatten()
    check_predicts(results, y, names, errors)
    tf.keras.backend.clear_session()
    count2, ds2 = morepork_support.build_dataset(recording_ids[count:], recording_samples)
    model = morepork_support.build_model(resnet_size, conv_size, conv_strides, max_pooling)
    training_steps = count2 // training_parameters.batch_size
    model.fit(ds2, steps_per_epoch=training_steps, epochs=1000, verbose=0)
    x, y, names = morepork_support.build_test(recording_ids[:count], recording_samples)
    results = model.predict(x).flatten()
    check_predicts(results, y, names, errors)
    tf.keras.backend.clear_session()
    print(f'completed pass {i} with {len(errors)} unique errors found')
    try:
        cur = db_connection.cursor()
        sql = ''' DELETE FROM training_errors '''
        cur.execute(sql)
        for name, count in errors.items():
            (id, span) = name.split('[')
            splits = span.split('-')
            start = float(splits[0])
            end = float(splits[1].split(']')[0])
            sql = ''' INSERT INTO training_errors(recording_id, start_time, end_time, number_of_times, checked)
                      VALUES(?, ?, ?, ?, 0)  '''
            cur.execute(sql, (int(id), start, end, count))
        db_connection.commit()
    except sqlite3.Error as e:
        print(e)