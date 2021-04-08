
import librosa
import numpy as np
import pickle
import tensorflow as tf

import sampler_dataset
import training_parameters

from resnet import ResnetBuilder

def scale(segments):
    for key in segments.keys():
        segment = segments[key]
        segment = librosa.amplitude_to_db(segment, ref=np.max)
        segments[key] = segment / abs(segment.min()) + 1.0

def gather_segments(ids, samples):
    segs = []
    vals = []
    names = []
    for id in ids:
        t = samples[id]
        for name, samp in t[0]:
            segs.append(samp)
            vals.append(1)
            names.append(name)
        for name, samp in t[1]:
            segs.append(samp)
            vals.append(0)
            names.append(name)
    return segs, vals, names

def build_dataset(ids, recording_samples):
    sample_dims = (training_parameters.num_buckets, training_parameters.slices_per_sample)
    segs, vals, _  = gather_segments(ids, recording_samples)
    sampler = sampler_dataset.DataSampler(segs, vals, sample_dims)
    ds = sampler.to_dataset()
    return len(segs), ds.shuffle(buffer_size=len(segs)).batch(training_parameters.batch_size).repeat()

def build_model(resnet_size, conv_size, conv_strides, max_pooling):
    input_dims = (training_parameters.num_buckets, training_parameters.slices_per_sample, 1)
    if resnet_size == 101:
        model = ResnetBuilder.build_resnet_101(input_dims, 1, conv_size, conv_strides, max_pooling)
    elif resnet_size == 50:
        model = ResnetBuilder.build_resnet_50(input_dims, 1, conv_size, conv_strides, max_pooling)
    elif resnet_size == 34:
        model = ResnetBuilder.build_resnet_34(input_dims, 1, conv_size, conv_strides, max_pooling)
    elif resnet_size == 18:
        model = ResnetBuilder.build_resnet_18(input_dims, 1, conv_size, conv_strides, max_pooling)
    optimizer = tf.keras.optimizers.Adam(lr=0.002, epsilon=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model

def build_test(ids, recording_samples):
    segs, vals, names  = gather_segments(ids, recording_samples)
    return np.array([sampler_dataset.extract_sample(seg) for seg in segs]), np.array(vals), np.array(names)

def load_samples():
    with open(training_parameters.segments_path, 'rb') as f:
        positive_segments = pickle.load(f)
        negative_segments = pickle.load(f)
        scale(positive_segments)
        scale(negative_segments)
    return positive_segments, negative_segments

def load_recording_samples():
    positive_segments, negative_segments = load_samples()
    recording_samples = {}
    
    def get_sample_lists(key):
        id = key.split('[')[0]
        if not id in recording_samples:
            recording_samples[id] = ([], [])
        return recording_samples[id]

    for item in positive_segments.items():
        samples = get_sample_lists(item[0])
        samples[0].append(item)
    for item in negative_segments.items():
        samples = get_sample_lists(item[0])
        samples[1].append(item)
    return recording_samples
