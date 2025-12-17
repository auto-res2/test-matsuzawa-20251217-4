"""
Model architecture implementations for D-RAdam experiment.
Implements ResNet-20 and DenseNet-40 for CIFAR-10/100.
"""

import torch
import torch.nn as nn
from typing import Optional
from omegaconf import DictConfig


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    
    expansion = 1
    
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet architecture for image classification."""
    
    def __init__(
        self,
        depth: int,
        num_classes: int,
        input_channels: int = 3,
    ):
        super(ResNet, self).__init__()
        
        if depth not in [20, 32, 44, 56, 110]:
            raise ValueError(f"Unsupported depth: {depth}")
        
        n = (depth - 2) // 6
        assert 6 * n + 2 == depth, f"Invalid depth for ResNet: {depth}"
        
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(
            input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        
        self._init_weights()
    
    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a residual layer with specified number of blocks."""
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        
        layers = []
        layers.append(
            BasicBlock(self.in_planes, planes, stride, downsample)
        )
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights using Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


class DenseBlock(nn.Module):
    """Dense block for DenseNet."""
    
    def __init__(self, in_channels: int, growth_rate: int = 12):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )
        
        self.growth_rate = growth_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = torch.cat([x, out], 1)
        return out


class DenseNet(nn.Module):
    """DenseNet architecture for image classification."""
    
    def __init__(
        self,
        depth: int,
        growth_rate: int,
        num_classes: int,
        input_channels: int = 3,
    ):
        super(DenseNet, self).__init__()
        
        assert (depth - 4) % 3 == 0, "Invalid depth for DenseNet"
        self.growth_rate = growth_rate
        
        blocks_per_layer = (depth - 4) // 3
        
        self.conv1 = nn.Conv2d(
            input_channels, 2 * growth_rate, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        
        num_channels = 2 * growth_rate
        
        self.dense1 = self._make_dense_layer(num_channels, blocks_per_layer)
        num_channels += blocks_per_layer * growth_rate
        self.transition1 = self._make_transition(num_channels)
        num_channels //= 2
        
        self.dense2 = self._make_dense_layer(num_channels, blocks_per_layer)
        num_channels += blocks_per_layer * growth_rate
        self.transition2 = self._make_transition(num_channels)
        num_channels //= 2
        
        self.dense3 = self._make_dense_layer(num_channels, blocks_per_layer)
        num_channels += blocks_per_layer * growth_rate
        
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)
        
        self._init_weights()
    
    def _make_dense_layer(
        self,
        in_channels: int,
        num_blocks: int,
    ) -> nn.Sequential:
        """Create dense layer with specified number of dense blocks."""
        layers = []
        for _ in range(num_blocks):
            layers.append(DenseBlock(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)
    
    def _make_transition(self, in_channels: int) -> nn.Sequential:
        """Create transition layer (bottleneck + average pooling)."""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        
        out = self.dense1(out)
        out = self.transition1(out)
        
        out = self.dense2(out)
        out = self.transition2(out)
        
        out = self.dense3(out)
        out = self.bn_final(out)
        out = nn.functional.relu(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def build_model(cfg: DictConfig) -> nn.Module:
    """Build model based on configuration."""
    model_name = cfg.name.lower()
    
    if "resnet" in model_name:
        if "20" in model_name:
            depth = 20
        elif "32" in model_name:
            depth = 32
        elif "44" in model_name:
            depth = 44
        elif "56" in model_name:
            depth = 56
        else:
            depth = cfg.get("depth", 20)
        
        return ResNet(
            depth=depth,
            num_classes=cfg.num_classes,
            input_channels=cfg.get("input_channels", 3),
        )
    
    elif "densenet" in model_name:
        if "40" in model_name:
            depth = 40
        else:
            depth = cfg.get("depth", 40)
        
        growth_rate = cfg.get("growth_rate", 12)
        
        return DenseNet(
            depth=depth,
            growth_rate=growth_rate,
            num_classes=cfg.num_classes,
            input_channels=cfg.get("input_channels", 3),
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
