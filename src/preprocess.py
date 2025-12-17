"""
Data loading and preprocessing pipeline for D-RAdam experiment.
Supports CIFAR-10, CIFAR-100 with proper caching and train/val/test separation.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_cifar10_transforms(augmentation: bool = True, normalize: bool = True) -> Tuple:
    """Get CIFAR-10 data transforms."""
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if normalize:
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        
        normalize_transform = transforms.Normalize(
            mean=cifar10_mean,
            std=cifar10_std,
        )
        
        train_transform = transforms.Compose(list(train_transform.transforms) + [normalize_transform])
        test_transform = transforms.Compose(list(test_transform.transforms) + [normalize_transform])
    
    return train_transform, test_transform


def get_cifar100_transforms(augmentation: bool = True, normalize: bool = True) -> Tuple:
    """Get CIFAR-100 data transforms."""
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if normalize:
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        
        normalize_transform = transforms.Normalize(
            mean=cifar100_mean,
            std=cifar100_std,
        )
        
        train_transform = transforms.Compose(list(train_transform.transforms) + [normalize_transform])
        test_transform = transforms.Compose(list(test_transform.transforms) + [normalize_transform])
    
    return train_transform, test_transform


def load_dataset(dataset_name: str, cfg: DictConfig) -> Dataset:
    """Load dataset based on configuration. Returns combined train+test dataset for later splitting."""
    dataset_name = dataset_name.lower()
    cache_dir = ".cache"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    if "cifar10" in dataset_name or "alpaca" in dataset_name:
        logger.info("Loading CIFAR-10 dataset")
        
        augmentation = cfg.preprocessing.get("augmentation", True)
        normalize = cfg.preprocessing.get("normalize", True)
        train_transform, test_transform = get_cifar10_transforms(augmentation, normalize)
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=cache_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=cache_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        return combined_dataset
    
    elif "cifar100" in dataset_name:
        logger.info("Loading CIFAR-100 dataset")
        
        augmentation = cfg.preprocessing.get("augmentation", True)
        normalize = cfg.preprocessing.get("normalize", True)
        train_transform, test_transform = get_cifar100_transforms(augmentation, normalize)
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=cache_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=cache_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        return combined_dataset
    
    else:
        logger.warning(f"Unknown dataset: {dataset_name}, using CIFAR-10")
        augmentation = cfg.preprocessing.get("augmentation", True)
        normalize = cfg.preprocessing.get("normalize", True)
        train_transform, test_transform = get_cifar10_transforms(augmentation, normalize)
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=cache_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=cache_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
        
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        return combined_dataset
