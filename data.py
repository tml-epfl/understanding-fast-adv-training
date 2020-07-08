import os
import torch
import torch.utils.data as td
import numpy as np
from torchvision import datasets, transforms


def get_loaders(dataset, n_ex, batch_size, train_set, shuffle, data_augm):
    dir_ = '~/data/' if os.path.exists('/home/maksym') else '/tmldata1/andriush/data'
    dataset_f = datasets_dict[dataset]
    num_workers = 2
    data_augm_transforms = [transforms.RandomCrop(32, padding=4)]
    if dataset not in ['mnist', 'svhn']:
        data_augm_transforms.append(transforms.RandomHorizontalFlip())
    transform_list = data_augm_transforms if data_augm else []
    transform = transforms.Compose(transform_list + [transforms.ToTensor()])

    if 'binary' in dataset:
        cl1, cl2 = 4, 8  # for cifar10 (4, 8) corresponds to deers vs ships
    if train_set:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=True, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='train', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex

        if 'binary' in dataset:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        if dataset == 'svhn':
            data.targets = data.labels
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                             num_workers=num_workers, drop_last=True)
    else:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=False, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='test', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex

        if 'binary' in dataset:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)  # to reduce memory consumption
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        if dataset == 'svhn':
            data.targets = data.labels
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                                             num_workers=2, drop_last=False)
    return loader


datasets_dict = {'mnist': datasets.MNIST, 'svhn': datasets.SVHN, 'cifar10': datasets.CIFAR10,
                 'cifar10_binary': datasets.CIFAR10, 'cifar10_binary_gs': datasets.CIFAR10
                 }
shapes_dict = {'mnist': (60000, 1, 28, 28), 'svhn': (73257, 3, 32, 32), 'cifar10': (50000, 3, 32, 32),
               'cifar10_binary': (10000, 3, 32, 32), 'cifar10_binary_gs': (10000, 1, 32, 32),
               'uniform_noise': (1000, 1, 28, 28)
               }
classes_dict = {'cifar10': {0: 'airplane',
                            1: 'automobile',
                            2: 'bird',
                            3: 'cat',
                            4: 'deer',
                            5: 'dog',
                            6: 'frog',
                            7: 'horse',
                            8: 'ship',
                            9: 'truck',
                            }
                }
