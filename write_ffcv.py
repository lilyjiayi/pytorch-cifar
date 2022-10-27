from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
import numpy as np
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
dataset = 'geo_yfcc'
root_dir = f'/nlp/scr-sync/nlp/{dataset}'
# root_dir = '/juice2/u/nlp/data/geo_yfcc'
my_dataset = get_dataset(dataset=dataset, download=False, root_dir = root_dir)
train_write_path = f'/nlp/scr-sync/nlp/geo_yfcc_ffcv/one_label_train.beton'
val_write_path = f'/nlp/scr-sync/nlp/geo_yfcc_ffcv/one_label_val.beton'
# write_path = '/juice2/u/nlp/data/geo_yfcc/one_label_test.beton'

train_data = my_dataset.get_subset("train")
train_data = WILDSSubset(train_data, range(47800,577364), transform=None)
val_data = my_dataset.get_subset("val")
# val_data = WILDSSubset(val_data, range(22000,144362), transform=None)

# Pass a type for each data field
train_writer = DatasetWriter(train_write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(max_resolution=500, jpeg_quality=90, compress_probability=1),
    'label': IntField(),
    'metadata': NDArrayField(shape=(3,), dtype=np.dtype('int64'))
    },
    num_workers=2)

# Write dataset
train_writer.from_indexed_dataset(train_data)

# # Pass a type for each data field
# val_writer = DatasetWriter(val_write_path, {
#     # Tune options to optimize dataset size, throughput at train-time
#     'image': RGBImageField(max_resolution=500, jpeg_quality=90, compress_probability=1),
#     'label': IntField(),
#     'metadata': NDArrayField(shape=(3,), dtype=np.dtype('int64')),
#     },
#     num_workers=2)

# # Write dataset
# val_writer.from_indexed_dataset(val_data)

