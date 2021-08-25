import torch
from torchvision.datasets import CIFAR100
import os
from imagenetv2_pytorch import ImageNetV2Dataset


def get_data_loader(preprocess_obj, data):
    print(f"Using dataset : {data}")
    classes = None
    if data == 'cifar100':
        images = CIFAR100(os.path.expanduser("~/.cache"),
                            transform=preprocess_obj,
                            download=True)
        loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=0)
        classes = images.classes
    elif data == 'imagenetv2':
        images = ImageNetV2Dataset(transform=preprocess_obj,
                                   location='../')
        loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=0)
    elif data == 'custom':
        raise NotImplemented("Custom data source not implemented yet")
    else:
        raise Exception("Please specify the data source")

    return loader, classes