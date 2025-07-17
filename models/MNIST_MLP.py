'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation MLP for MNIST Model
******************************************************************************
'''

import torch.nn as nn
import torch.nn.functional as F

from modules import LinearAsGNN


# Define the MLP model using LinearAsGNN
class MLPAsGNNMNIST(nn.Module):
    def __init__(self):
        super(MLPAsGNNMNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = LinearAsGNN(28 * 28, 250)
        self.linear2 = LinearAsGNN(250, 100)
        self.linear3 = LinearAsGNN(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = F.log_softmax(self.linear3(x), dim=-1)
        return y_pred

# Define a simple MLP model using nn.Linear
class MLPMNIST(nn.Module):
    def __init__(self):
        super(MLPMNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = F.log_softmax(self.linear3(x), dim=-1)
        return y_pred