
import datetime
import librosa
import os
import numpy as np
import pickle
import random
from random import randrange
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from training.resnet import ResnetBuilder
from training import training_parameters

new_load = True
trainfract = .8
resnet_size = 34
conv_size = (7,7)
conv_strides = (2,2)
max_pooling = True
if max_pooling:
    pooling = 'max'
else:
    pooling = 'avg'
save_path = f'{training_parameters.base_path}/morepork-resnet{resnet_size}-6248-{conv_size[0]}-{conv_size[1]}-{conv_strides[0]}-{conv_strides[1]}-{pooling}'
if not os.path.exists(save_path):
    os.mkdir(save_path)
print(f'Saving results to directory {save_path}')


def load_segments(new_shuffle):
    def scale(segments):
        for key in segments.keys():
            segment = segments[key]
            segment = librosa.amplitude_to_db(segment, ref=np.max)
            segments[key] = segment / abs(segment.min()) + 1.0

    with open(training_parameters.segments_path, 'rb') as f:
        positive_segments = pickle.load(f)
        negative_segments = pickle.load(f)
    scale(positive_segments)
    scale(negative_segments)
    if new_shuffle:
        positive_names = list(positive_segments.keys())
        random.shuffle(positive_names)
        with open(f'{training_parameters.base_path}/shuffled-positives.txt', 'wb') as f:
            np.save(f, positive_names)
        negative_names = list(negative_segments.keys())
        random.shuffle(negative_names)
        with open(f'{training_parameters.base_path}/shuffled-negatives.txt', 'wb') as f:
            np.save(f, negative_names)
    else:
        with open(f'{training_parameters.base_path}/shuffled-positives.txt', 'rb') as f:
            positive_names = list(np.load(f))
        with open(f'{training_parameters.base_path}/shuffled-negatives.txt', 'rb') as f:
            negative_names = list(np.load(f))
    return positive_segments, negative_segments, positive_names, negative_names


# load all the samples
positive_segments, negative_segments, positive_names, negative_names = load_segments(new_load)


def get_segments(names, segments):
    segment_list = []
    for name in names:
        segment_list.append(segments[name])
    return segment_list

# split samples into train and validate
positive_training_count = int(len(positive_names) * trainfract)
negative_training_count = int(len(negative_names) * trainfract)
train_morepork_names = positive_names[:positive_training_count]
train_moreporks = get_segments(train_morepork_names, positive_segments)
train_notpork_names = negative_names[:negative_training_count]
train_notporks = get_segments(train_notpork_names, negative_segments)
validate_morepork_names = positive_names[positive_training_count:]
validate_notpork_names = negative_names[negative_training_count:]
validate_segments = get_segments(validate_morepork_names, positive_segments) + get_segments(validate_notpork_names, negative_segments)
validate_names = validate_morepork_names + validate_notpork_names
validate_actuals = [1] * len(validate_morepork_names) + [0] * len(validate_notpork_names)

print(f'Training with base {len(train_moreporks)} positive samples and {len(train_notporks)} negative samples')
print(f'Validating with {len(validate_morepork_names)} positive samples and {len(validate_notpork_names)} negative samples')

train_moreporks_count = len(train_moreporks)
train_notporks_count = len(train_notporks)

x_train = train_moreporks+train_notporks
y_train = [1]*len(train_moreporks)+[0]*len(train_notporks)
print(len(y_train))

class DataSampler:

    def __init__(self, x, y, dims):
        self.x = x
        self.y = y
        self.dims = dims

    def sampler(self):
        i = 0
        while i < len(self.x):
            input = self.x[i]
            num_extra_slices = input.shape[1] - self.dims[1]
            if num_extra_slices:
                base_column = randrange(0, num_extra_slices+1)
            else:
                base_column = 0
            num_extra_rows = input.shape[0] - self.dims[0]
            if num_extra_rows:
                base_row = randrange(0, num_extra_rows+1)
            else:
                base_row = 0
            sample = input[base_row:base_row+self.dims[0], base_column:base_column + self.dims[1]].copy()
            sample = sample / sample.max()
            sample = sample.reshape(sample.shape + (1,))
            yield (sample,self.y[i])
            i += 1

    def to_dataset(self):
        shape_list = list(self.dims)
        shape_list.append(None)
        out_types = (tf.float32, tf.float32)
        out_shapes = (tuple(shape_list), ())
        return tf.data.Dataset.from_generator(self.sampler, output_types=out_types, output_shapes=out_shapes)

sample_dims = (training_parameters.num_buckets, training_parameters.slices_per_sample)
training_sampler = DataSampler(x_train, y_train, sample_dims)
training_ds = training_sampler.to_dataset()
training_ds = training_ds.shuffle(buffer_size=len(x_train)).batch(training_parameters.batch_size).repeat()
input_dims = (training_parameters.num_buckets, training_parameters.slices_per_sample, 1)
if resnet_size == 101:
    model = ResnetBuilder.build_resnet_101(input_dims, 1, conv_size, conv_strides, max_pooling)
elif resnet_size == 50:
    model = ResnetBuilder.build_resnet_50(input_dims, 1, conv_size, conv_strides, max_pooling)
elif resnet_size == 34:
    model = ResnetBuilder.build_resnet_34(input_dims, 1, conv_size, conv_strides, max_pooling)
elif resnet_size == 18:
    model = ResnetBuilder.build_resnet_18(input_dims, 1, conv_size, conv_strides, max_pooling)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
with open(f'{save_path}/model.txt', 'w') as f:

    def summary_print(s):
        print(s, file=f)

    print(model.summary(print_fn=summary_print))

# serialize model to JSON
model_json = model.to_json()
with open(f'{save_path}/morepork-model.json', 'w') as f:
    f.write(model_json)

validation_sampler = DataSampler(validate_segments, validate_actuals, sample_dims)
validation_ds = validation_sampler.to_dataset()
validation_ds = validation_ds.shuffle(buffer_size=len(validate_segments)).batch(training_parameters.batch_size).repeat()
training_steps = (train_moreporks_count+train_notporks_count) // training_parameters.batch_size
validation_steps = len(validate_segments) // training_parameters.batch_size
log_dir = f'{training_parameters.base_path}/logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=10, write_graph=False)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=0.0001)
checkpoint_callback = ModelCheckpoint(filepath=save_path+'/weights-{epoch:02d}-{val_acc:.4f}.h5',
    save_weights_only=True, monitor='val_acc', mode='max', save_best_only=True)

results = model.fit(training_ds, steps_per_epoch=training_steps,
                    validation_data=validation_ds, validation_steps=validation_steps,
                    callbacks=[tensorboard_callback, reduce_lr_callback, checkpoint_callback], epochs=3000)
