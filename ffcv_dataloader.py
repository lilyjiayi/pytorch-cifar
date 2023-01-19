from configparser import NoSectionError
from pathlib import Path
import torch
import numpy as np

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip, NormalizeImage, Squeeze


train_path = {
    'geo_yfcc' : '/self/scr-sync/nlp/geo_yfcc_ffcv/one_label_train.beton',
    'imagenet' : '/self/scr-sync/nlp/imagenet_ffcv/train_512_1.0_90.ffcv',
    'yfcc_imagenet' : '/self/scr-sync/nlp/yfcc_imagenet_ffcv/one_label_all.beton',
    'combined_imagenet': '/nlp/scr-sync/nlp/combined_imagenet_ffcv/combined.beton',
}

val_path = {
    'geo_yfcc' : '/self/scr-sync/nlp/geo_yfcc_ffcv/one_label_val.beton',
    'imagenet' : '/self/scr-sync/nlp/imagenet_ffcv/val_512_1.0_90.ffcv',
    'yfcc_imagenet' : '/self/scr-sync/nlp/yfcc_imagenet_ffcv/one_label_all.beton',
    'combined_imagenet' : '/nlp/scr-sync/nlp/combined_imagenet_ffcv/combined.beton',
}


def ffcv_train_loader(dataset_name, dataset=None, indices=None, num_workers=8, batch_size=128, pin_memory=True, drop_last=True):
    path = train_path[dataset_name]
    if dataset_name == 'geo_yfcc':
        loader = yfcc_train_loader(path, indices, num_workers, batch_size, pin_memory, drop_last)
    elif dataset_name == 'imagenet' or dataset_name == 'yfcc_imagenet':
        loader = imagenet_train_loader(path, indices, num_workers, batch_size, pin_memory, drop_last)
    elif dataset_name == 'combined_imagenet':
        loader = imagenet_train_loader2(path, indices, num_workers, batch_size, pin_memory, drop_last)
        # imagenet_indices, yfcc_imagenet_indices, imagenet_part, yfcc_imagenet_part = get_separate_indices(dataset, indices)
        # imagenet_loader = imagenet_train_loader(train_path["imagenet"], imagenet_indices, num_workers, batch_size, pin_memory, drop_last)
        # yfcc_imagenet_loader = imagenet_train_loader(train_path["yfcc_imagenet"], yfcc_imagenet_indices, num_workers, batch_size, pin_memory, drop_last)
        # loader = CombinedLoader(imagenet_part, yfcc_imagenet_part, imagenet_loader, yfcc_imagenet_loader)
    return loader

def ffcv_val_loader(dataset_name, dataset=None, indices=None, num_workers=8, batch_size=128, pin_memory=True):
    path = val_path[dataset_name]
    if dataset_name == 'geo_yfcc':
        loader = yfcc_val_loader(path, indices, num_workers, batch_size, pin_memory)
    elif dataset_name == 'imagenet' or dataset_name == 'yfcc_imagenet':
        loader = imagenet_val_loader(path, indices, num_workers, batch_size, pin_memory)
    elif dataset_name == 'combined_imagenet':
        loader = imagenet_val_loader2(path, indices, num_workers, batch_size, pin_memory)
        # imagenet_indices, yfcc_imagenet_indices, imagenet_part, yfcc_imagenet_part = get_separate_indices(dataset, indices)
        # imagenet_loader = imagenet_val_loader(train_path["imagenet"], imagenet_indices, num_workers, batch_size, pin_memory)
        # yfcc_imagenet_loader = imagenet_val_loader(train_path["yfcc_imagenet"], yfcc_imagenet_indices, num_workers, batch_size, pin_memory)
        # loader = CombinedLoader(imagenet_part, yfcc_imagenet_part, imagenet_loader, yfcc_imagenet_loader)
    return loader

def ffcv_train_val_loader(dataset_name, dataset=None, indices=None, num_workers=8, batch_size=128, pin_memory=True):
    path = train_path[dataset_name]
    if dataset_name == 'geo_yfcc':
        loader = yfcc_val_loader(path, indices, num_workers, batch_size, pin_memory)
    elif dataset_name == 'imagenet' or dataset_name == 'yfcc_imagenet':
        loader = imagenet_val_loader(path, indices, num_workers, batch_size, pin_memory)
    elif dataset_name == 'combined_imagenet':
         loader = imagenet_val_loader2(path, indices, num_workers, batch_size, pin_memory)
        # imagenet_indices, yfcc_imagenet_indices, imagenet_part, yfcc_imagenet_part = get_separate_indices(dataset, indices)
        # imagenet_loader = imagenet_val_loader(train_path["imagenet"], imagenet_indices, num_workers, batch_size, pin_memory)
        # yfcc_imagenet_loader = imagenet_val_loader(train_path["yfcc_imagenet"], yfcc_imagenet_indices, num_workers, batch_size, pin_memory)
        # loader = CombinedLoader(imagenet_part, yfcc_imagenet_part, imagenet_loader, yfcc_imagenet_loader)
    return loader

label_pipeline = [#List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(f'cuda:{0}'), non_blocking=True)
]

def imagenet_train_loader(path, indices=None, num_workers=8, batch_size=128, pin_memory=True, drop_last=True):
    # path = '/self/scr-sync/nlp/imagenet_ffcv/train_512_1.0_90.ffcv'
    target_resolution = (224, 224)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    this_device = f'cuda:{0}'
    # decoder = RandomResizedCropRGBImageDecoder((160,160))
    decoder = RandomResizedCropRGBImageDecoder(target_resolution)
    
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    train_loader = Loader(path,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.QUASI_RANDOM,
                        drop_last=drop_last,
                        indices=indices,
                        os_cache=pin_memory,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
    })
    return train_loader

def imagenet_val_loader(path, indices=None, num_workers=8, batch_size=128, pin_memory=True):
    # path = '/self/scr-sync/nlp/imagenet_ffcv/val_512_1.0_90.ffcv'
    target_resolution = (224, 224)
    crop_ratio = 224.0/256.0
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    this_device = f'cuda:{0}'
    cropper = CenterCropRGBImageDecoder(target_resolution, ratio=crop_ratio)

    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    val_loader = Loader(path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            indices=indices,
            os_cache=pin_memory,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
    })
    return val_loader

def imagenet_train_loader2(path, indices=None, num_workers=8, batch_size=128, pin_memory=True, drop_last=True):
    # path = '/self/scr-sync/nlp/imagenet_ffcv/train_512_1.0_90.ffcv'
    target_resolution = (224, 224)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    this_device = f'cuda:{0}'
    # decoder = RandomResizedCropRGBImageDecoder((160,160))
    decoder = RandomResizedCropRGBImageDecoder(target_resolution)
    
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    train_loader = Loader(path,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.QUASI_RANDOM,
                        drop_last=drop_last,
                        indices=indices,
                        os_cache=pin_memory,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline,
                            'metadata': [NDArrayDecoder(), ToTensor()]
    })
    return train_loader

def imagenet_val_loader2(path, indices=None, num_workers=8, batch_size=128, pin_memory=True):
    # path = '/self/scr-sync/nlp/imagenet_ffcv/val_512_1.0_90.ffcv'
    target_resolution = (224, 224)
    crop_ratio = 224.0/256.0
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    this_device = f'cuda:{0}'
    cropper = CenterCropRGBImageDecoder(target_resolution, ratio=crop_ratio)

    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    val_loader = Loader(path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            indices=indices,
            os_cache=pin_memory,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline,
                'metadata': [NDArrayDecoder(), ToTensor()]
    })
    return val_loader

def yfcc_train_loader(path, indices=None, num_workers=8, batch_size=128, pin_memory=True, drop_last=True):
    # path = '/self/scr-sync/nlp/geo_yfcc_ffcv/one_label_train.beton'
    # path = '/self/scr-sync/nlp/geo_yfcc_ffcv_jpg/one_label_all.beton'
    target_resolution = (224, 224)
    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder(target_resolution, scale=(0.9, 1.0), ratio=(0.75, 1.3333333333333333)),
        RandomHorizontalFlip(),
        NormalizeImage(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]), np.dtype('float32')),
        ToTensor(),
        ToTorchImage(),
    ]

    train_loader = Loader(path,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.RANDOM,
                        drop_last=drop_last,
                        indices=indices,
                        os_cache=pin_memory,
                        pipelines={
                            'image': train_image_pipeline,
                            'label': label_pipeline,
                            'metadata': [NDArrayDecoder(), ToTensor()]
    })
    return train_loader

def yfcc_val_loader(path, indices=None, num_workers=8, batch_size=128, pin_memory=True):
    # path = '/self/scr-sync/nlp/geo_yfcc_ffcv/one_label_val.beton'
    # path = '/self/scr-sync/nlp/geo_yfcc_ffcv_jpg/one_label_all.beton'
    target_resolution = (224, 224)
    eval_image_pipeline = [
        CenterCropRGBImageDecoder(target_resolution, 224.0/256.0),
        NormalizeImage(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]), np.dtype('float32')),
        ToTensor(),
        ToTorchImage(),
    ]
    
    val_loader = Loader(path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            indices=indices,
            os_cache=pin_memory,
            pipelines={
                'image': eval_image_pipeline,
                'label': label_pipeline,
                'metadata': [NDArrayDecoder(), ToTensor()]
    })
    return val_loader
class CombinedLoader:
    def __init__(self, imagenet_part, yfcc_imagenet_part, imagenet_loader, yfcc_loader):
        rearranged = np.concatenate((imagenet_part, yfcc_imagenet_part))
        loaders = [imagenet_loader, yfcc_loader]
        # rearranged = np.concatenate((yfcc_imagenet_part, imagenet_part))
        # loaders = [yfcc_loader, imagenet_loader]
        self.indices = np.argsort(rearranged)
        self.loaders = loaders
        list_idx = []
        self.length = 0
        for loader in self.loaders:
            self.length += len(loader)
            list_idx.append(loader.indices)
        # self.indices = np.concatenate(list_idx)
    
    def __len__(self):
        return self.length

    def __iter__(self):
        def generator(loaders):
            for loader in loaders:
                for batch in loader:
                    yield batch
        return generator(self.loaders)


def get_separate_indices(dataset, indices):
    if indices is None:
        indices = np.arange(len(dataset))
    selected = dataset.metadata_array.numpy()[indices]
    dataset_array = selected[:,0]
    imagenet_part = np.nonzero(dataset_array == 0)[0]
    yfcc_imagenet_part = np.nonzero(dataset_array == 1)[0]
    rearranged = np.concatenate((imagenet_part, yfcc_imagenet_part))
    rearranged_indices = np.argsort(rearranged)
    return selected[imagenet_part, 1], selected[yfcc_imagenet_part, 1], imagenet_part, yfcc_imagenet_part
    