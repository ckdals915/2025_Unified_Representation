'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation ResNet as GNN
******************************************************************************
'''

import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models

from modules import Conv2dAsGNN
from modules import MaxPool2dAsGNN, AvgPool2dAsGNN

# A basic residual block using GNN-based convolution and optional downsampling.
class BasicBlockAsGNN(nn.Module):
    # BasicBlockAsGNN is a basic residual block for ResNet architectures.
    expansion = 1
    
    
    def __init__(self, height, width, in_channels, out_channels, stride=1, bias=False):
        super(BasicBlockAsGNN, self).__init__()

        # First 3x3 convolution
        self.conv1 = Conv2dAsGNN(
            height=height,
            width=width,
            kernel_size=3,
            stride=stride,
            in_channels=in_channels,
            out_channels=out_channels,
            padding='same',
            bias=bias
        )
        
        # Batch normalization after first convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        h1, w1 = self.conv1.out_height, self.conv1.out_width

        # Second 3x3 convolution
        self.conv2 = Conv2dAsGNN(
            height=h1,
            width=w1,
            kernel_size=3,
            stride=1,
            in_channels=out_channels,
            out_channels=out_channels,
            padding='same',
            bias=bias
        )
        
        # Batch normalization after second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Downsample if shape or channel mismatch
        self.downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlockAsGNN.expansion:
            self.downsample = nn.Sequential(
                Conv2dAsGNN(
                    height=height,
                    width=width,
                    kernel_size=1,
                    stride=stride,
                    in_channels=in_channels,
                    out_channels=out_channels * BasicBlockAsGNN.expansion,
                    padding='valid',
                    bias=bias
                ),
                nn.BatchNorm2d(out_channels * BasicBlockAsGNN.expansion)
            )

        # Store output dims
        self.out_height, self.out_width = self.conv2.out_height, self.conv2.out_width

    def forward(self, x):
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

# ResNet-18 architecture using GNN-based convolution and pooling layers.
class ResNetAsGNNCIFAR100(nn.Module):
    def __init__(self, height=32, width=32, num_classes=1000, bias=False):
        super(ResNetAsGNNCIFAR100, self).__init__()
        
        # Initial in_channels
        self.in_channels = 64

        # Convolutional layer for initial feature extraction
        self.conv1 = Conv2dAsGNN(
            height=height,
            width=width,
            kernel_size=7,
            stride=2,
            in_channels=3,
            out_channels=self.in_channels,
            padding='same',
            bias=bias
        )
        
        # Batch normalization and ReLU activation
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Height and width update
        h, w = self.conv1.out_height, self.conv1.out_width

        # Max pooling layer
        self.maxpool = MaxPool2dAsGNN(
            height=h,
            width=w,
            kernel_size=3,
            stride=2,
            padding='same'
        )
        
        # Update height and width after pooling
        h, w = self.maxpool.out_height, self.maxpool.out_width

        # Residual layers
        self.layer1 = self._make_layer(h, w, 64, blocks=2, stride=1)
        h, w = self.layer1[-1].out_height, self.layer1[-1].out_width
        self.layer2 = self._make_layer(h, w, 128, blocks=2, stride=2)
        h, w = self.layer2[-1].out_height, self.layer2[-1].out_width
        self.layer3 = self._make_layer(h, w, 256, blocks=2, stride=2)
        h, w = self.layer3[-1].out_height, self.layer3[-1].out_width
        self.layer4 = self._make_layer(h, w, 512, blocks=2, stride=2)
        h, w = self.layer4[-1].out_height, self.layer4[-1].out_width

        # Global average pooling and fully connected
        self.avgpool = AvgPool2dAsGNN(
            height=h,
            width=w,
            kernel_size=1,
            stride=1
        )
        self.fc = nn.Linear(512 * BasicBlockAsGNN.expansion, num_classes)

    def _make_layer(self, height, width, out_channels, blocks, stride):
        layers = []
        # First block downsample
        layers.append(
            BasicBlockAsGNN(
                height=height,
                width=width,
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride
            )
        )
        self.in_channels = out_channels * BasicBlockAsGNN.expansion

        # Remaining blocks
        for _ in range(1, blocks):
            h, w = layers[-1].out_height, layers[-1].out_width
            layers.append(
                BasicBlockAsGNN(
                    height=h,
                    width=w,
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Basic Block for resnet 18 and resnet 34
class BasicBlock(nn.Module):
    # Expansion factor for the block
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# BottleNeck block for resnet 50, resnet 101, resnet 152
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def ResNet18CIFAR100(num_classes=100):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def ResNet18AsGNNCIFAR100():
    return ResNetAsGNNCIFAR100(height=32, width=32, num_classes=100)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    model_resnet18 = ResNet18CIFAR100().to(device)
    summary(model_resnet18, (4, 3, 32, 32), device=device)

    model = ResNet18AsGNNCIFAR100(height=32, width=32, num_classes=100).to(device)
    summary(model, (4, 3, 32, 32), device=device)