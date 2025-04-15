import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import os
import json
import pickle

def load_dataset(name, mod, transform=None, adv=False, root_dir='./data2'):

    if name == 'mnist':
        if os.path.exists(os.path.join(root_dir, 'MNIST')):
            return datasets.MNIST(root_dir, train=(mod=='train'), transform=transform)
        return datasets.MNIST(root_dir, train=(mod=='train'), transform=transform, download=True)
    elif name == 'fmnist':
        if os.path.exists(os.path.join(root_dir, 'Fashion-MNIST')):
            return datasets.FashionMNIST(root_dir, train=(mod=='train'), transform=transform)
        return datasets.FashionMNIST(root_dir, train=(mod=='train'), transform=transform, download=True)
    elif name == 'cifar10':
        if os.path.exists(os.path.join(root_dir, 'cifar-10-batches-py')):
            return datasets.CIFAR10(root=root_dir, train=(mod=='train'), transform=transform,download=False)
        else:
            return datasets.CIFAR10(root_dir, train=(mod=='train'), transform=transform, download=True)
    elif name == 'cifar100':
        if os.path.exists(os.path.join(root_dir, 'cifar-100-batches-py')):
            return datasets.CIFAR100(root_dir, train=(mod=='train'), transform=transform)
        return datasets.CIFAR100(root_dir, train=(mod=='train'), transform=transform, download=True)
    elif name == 'tinyimagenet':
        root_dir='/home/zhang/E/robust_test/ZSRobust4FoundationModel-main/data'
        if os.path.exists(os.path.join(root_dir, 'tiny-imagenet-200')):
            if mod=='train':
                tinydataset = datasets.ImageFolder(os.path.join(root_dir,'tiny-imagenet-200', mod),transform=transform)
            elif mod=='test':
                tinydataset = datasets.ImageFolder(os.path.join(root_dir, 'tiny-imagenet-200', 'val'), transform=transform)
            return tinydataset
        return  False, f"Folder does not exist."

def load_transform(name, mod):
    if mod == 'train':    
        if name == 'cifar10' or name == 'cifar100':
            return transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        elif name =='tinyimagenet':
            return transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        else:
            return transforms.Compose([transforms.ToTensor()])
    elif mod == 'test':
        return transforms.Compose([transforms.ToTensor()])

class RandomNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomScale(object):
    def __init__(self, a=0.8, b=1):
        self.a, self.b = a, b
        
    def __call__(self, tensor):
        scale = self.a + torch.rand(tensor.shape[0]) * (self.b-self.a)
        return (tensor * scale).clamp(0,1)
        bias = torch.rand(tensor.shape[0]) * (1 - scale)
        return tensor * scale + bias
    
    def __repr__(self):
        return self.__class__.__name__ + '(a={0}, b={1})'.format(self.a, self.b)
