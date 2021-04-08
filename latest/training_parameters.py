
# static configuration; changing these parameters breaks model compatibility
slices_per_second = 10
num_buckets = 60
sample_window_size = 3
slices_per_sample = slices_per_second * sample_window_size * 2

# system configuration
base_path = '/hdd1/dennis'
segments_path = f'{base_path}/training-samples-66-570-1230.pk'
early_termination_patience = None
batch_size = 256