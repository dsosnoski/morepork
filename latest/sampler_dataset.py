
from random import randrange
import tensorflow as tf

import training_parameters


def extract_sample(input):
    num_extra_slices = input.shape[1] - training_parameters.slices_per_sample
    if num_extra_slices:
        base_column = randrange(0, num_extra_slices + 1)
    else:
        base_column = 0
    num_extra_rows = input.shape[0] - training_parameters.num_buckets
    if num_extra_rows:
        base_row = randrange(0, num_extra_rows + 1)
    else:
        base_row = 0
    sample = input[base_row:base_row + training_parameters.num_buckets, base_column:base_column + training_parameters.slices_per_sample].copy()
    sample = sample / sample.max()
    sample = sample.reshape(sample.shape + (1,))
    return sample


class DataSampler:

    def __init__(self, x, y, dims):
        self.x = x
        self.y = y
        self.dims = dims

    def sampler(self):
        i = 0
        while i < len(self.x):
            sample = extract_sample(self.x[i])
            yield (sample, self.y[i])
            i += 1

    def to_dataset(self):
        shape_list = list(self.dims)
        shape_list.append(None)
        out_types = (tf.float32, tf.float32)
        out_shapes = (tuple(shape_list), ())
        return tf.data.Dataset.from_generator(self.sampler, output_types=out_types, output_shapes=out_shapes)
