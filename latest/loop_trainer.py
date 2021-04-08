
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import morepork_support
from resnet import ResnetBuilder
from sampler_dataset import DataSampler
import training_parameters

train_fraction = .8
trainings_count = 20
resnet_size = 34
conv_size = (7,7)
conv_strides = (2,2)
max_pooling = True
max_noise = None
fixed_permuted = False
use_permuted = '/hdd1/dennis/morepork-resnet34-7-7-2-2-max-randomtestval1/weights1/permute.pk'
if max_pooling:
    pooling = 'max'
else:
    pooling = 'avg'
model_builder = {
    18: ResnetBuilder.build_resnet_18, 34: ResnetBuilder.build_resnet_34, 50: ResnetBuilder.build_resnet_50,
    101: ResnetBuilder.build_resnet_101, 152: ResnetBuilder.build_resnet_152
}[resnet_size]


class Checkpointer(ModelCheckpoint):

    def __init__(self, path, start, decay, rate):
        super().__init__(filepath=path, monitor='val_binary_accuracy', mode='max', save_best_only=True)
        self.start_epoch = start
        self.decay_time = decay
        self.decay_rate = rate
        self.save_active = False
        self.last_decay = 0
        self.monitor_op = self._check
        self.last_save = 0.0

    def _check(self, current, best):
        if self._current_epoch >= self.start_epoch:
            self.save_active = True
        if current > best:
            self.best = current
            self.last_save = current
            return self.save_active
        elif self.save_active and self.epochs_since_last_save >= self.decay_time:
            if self.epochs_since_last_save % self.decay_time == 0:
                self.best *= self.decay_rate

    def last_save_accuracy(self):
        return self.last_save

def build_model(conv_size, conv_strides, input_dims):
    model = model_builder(input_dims, 1, conv_size, conv_strides)
    optimizer = tf.keras.optimizers.Adam(lr=0.002, epsilon=0.002)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def first_time_model(model):
    print(model.summary())
    with open(f'{save_directory}/model.txt', 'w') as f:
        def summary_print(s):
            print(s, file=f)

        print(model.summary(print_fn=summary_print))
    model_json = model.to_json()
    with open(f'{save_directory}/model.json', 'w') as f:
        f.write(model_json)

positive_segments, negative_segments = morepork_support.load_samples()
segments = list(positive_segments.values()) + list(negative_segments.values())
actuals = [1.0] * len(positive_segments) + [0.0] * len(negative_segments)
validation_count = int(len(segments) * (1 - train_fraction) //
                       training_parameters.batch_size) * training_parameters.batch_size
train_count = len(segments) - validation_count
sample_dims = (training_parameters.num_buckets, training_parameters.slices_per_sample)
random_generator = np.random.default_rng()
save_directory = f'{training_parameters.base_path}/morepork-resnet{resnet_size}-{conv_size[0]}-{conv_size[1]}-{conv_strides[0]}-{conv_strides[1]}-{pooling}-randomtestval2'
if not os.path.exists(save_directory):
    os.mkdir(save_directory)
print(f'training with {train_count} samples, validating with {validation_count} samples, saving to {save_directory}')
sum_validation_accuracies = 0.0
if fixed_permuted:
    if use_permuted:
        with open(use_permuted, 'rb') as f:
            permuted = pickle.load(f)
    else:
        permuted = np.random.permutation(len(segments))
for i in range(trainings_count):
    save_path = f'{save_directory}/weights{i}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not fixed_permuted:
        permuted = np.random.permutation(len(segments))
        with open(f'{save_path}/permute.pk', 'wb') as f:
            pickle.dump(permuted, f)
    permuted_segments = [segments[i] for i in permuted]
    permuted_actuals = [actuals[i] for i in permuted]
    training_sampler = DataSampler(permuted_segments[:train_count], permuted_actuals[:train_count], sample_dims)
    training_ds = training_sampler.to_dataset()
    training_ds = training_ds.shuffle(buffer_size=train_count).batch(training_parameters.batch_size).repeat()
    validation_sampler = DataSampler(permuted_segments[train_count:], permuted_actuals[train_count:], sample_dims)
    validation_ds = validation_sampler.to_dataset()
    validation_ds = validation_ds.shuffle(buffer_size=validation_count).batch(
        training_parameters.batch_size).repeat()
    training_steps = train_count // training_parameters.batch_size
    validation_steps = validation_count // training_parameters.batch_size
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.65, patience=25, min_lr=0.0002, cooldown=25, verbose=1)
    checkpoint_callback = Checkpointer(save_path + '/model-{epoch:02d}-{val_binary_accuracy:.4f}', 200, 100, .9999)
    input_dims = (training_parameters.num_buckets, training_parameters.slices_per_sample, 1)
    model = build_model(conv_size, conv_strides, input_dims)
    if i == 0:
        first_time_model(model)
    history = model.fit(training_ds, steps_per_epoch=training_steps,
                        validation_data=validation_ds, validation_steps=validation_steps,
                        callbacks=[reduce_lr_callback, checkpoint_callback], epochs=1000)
    sum_validation_accuracies += checkpoint_callback.last_save_accuracy()
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Binary Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')

    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig(f'{save_path}/history.png')
    plt.close()

    del model
    tf.keras.backend.clear_session()

print(f'Average best validation accuracy {sum_validation_accuracies/trainings_count:04f}')