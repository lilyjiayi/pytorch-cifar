from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
import numpy as np
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSSubset

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
dataset = 'geo_yfcc'
root_dir = f'/nlp/scr-sync/nlp/{dataset}'
my_dataset = get_dataset(dataset=dataset, download=False, root_dir = root_dir)
write_path = '/scr-ssd/rtaori/one_label_all_final.beton'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': RGBImageField(write_mode='jpg', max_resolution=500, jpeg_quality=90, compress_probability=1),
    'label': IntField(),
    'metadata': NDArrayField(shape=(3,), dtype=np.dtype('int64'))
    },
    num_workers=8)

# Write dataset
writer.from_indexed_dataset(my_dataset, chunksize=100)
