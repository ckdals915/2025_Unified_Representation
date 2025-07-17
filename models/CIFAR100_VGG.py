'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation VGG as GNN
******************************************************************************
'''

import math
import torch.nn as nn

from modules import LinearAsGNN
from modules import Conv2dAsGNN
from modules import MaxPool2dAsGNN, AvgPool2dAsGNN


# VGG configuration dictionary
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGGCIFAR100(nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def VGG11CIFAR100():
    return VGGCIFAR100(make_layers(cfg['A'], batch_norm=True))

def make_layers_gnn(cfg_list, height, width, batch_norm=False):
    layers = []
    in_ch = 3
    H, W = height, width

    for v in cfg_list:
        if v == 'M':
            # Max pooling
            layers.append(
                MaxPool2dAsGNN(
                    height=H, width=W,
                    kernel_size=2, stride=2,
                    padding='valid'
                )
            )
            # H and W update
            H = math.floor((H - 2) / 2) + 1
            W = math.floor((W - 2) / 2) + 1

        else:
            # Convolutional layer
            layers.append(
                Conv2dAsGNN(
                    height=H, width=W,
                    kernel_size=3, stride=1,
                    in_channels=in_ch, out_channels=v,
                    padding='same', bias=True
                )
            )
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_ch = v

    return nn.Sequential(*layers), (H, W)


class VGGAsGNNCIFAR100(nn.Module):
    def __init__(self,
                 cfg_key: str = 'A',
                 input_size: tuple = (32, 32),
                 num_classes: int = 100,
                 batch_norm: bool = True):
        super().__init__()
        cfg_list = cfg[cfg_key]
        H0, W0 = input_size

        # Feature Block
        self.features, (Hf, Wf) = make_layers_gnn(
            cfg_list, height=H0, width=W0, batch_norm=batch_norm
        )

        # Classifier Block
        feat_dim = 512 * Hf * Wf
        self.classifier = nn.Sequential(
            LinearAsGNN(in_channels=feat_dim, out_channels=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LinearAsGNN(in_channels=4096, out_channels=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            LinearAsGNN(in_channels=4096, out_channels=num_classes)
        )

    def forward(self, x):
        x = self.features(x)               
        x = x.view(x.size(0), -1)    
        x = self.classifier(x)           
        return x          


def VGG11AsGNNCIFAR100():
    return VGGAsGNNCIFAR100(cfg_key='A', input_size=(32, 32), num_classes=100, batch_norm=True)
