'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation 1D-CNN for CWRU Dataset
******************************************************************************
'''

import torch
import torch.nn as nn
import math
from torchinfo import summary

from modules import LinearAsGNN
from modules import Conv1dAsGNN
from modules import MaxPool1dAsGNN


class CNN1DCWRU(torch.nn.Module):
    """
    A simple 1D CNN model for CWRU dataset.
    """
    def __init__(self):
        super(CNN1DCWRU, self).__init__()
        
        # Define the convolutional layers and fully connected layers
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),

            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv(x)         
        x = x.view(-1, 1024)
        x = self.fc(x)    
        return torch.nn.functional.softmax(x, dim=1)

class CNN1DCWRUAsGNN(nn.Module):
    """
    A 1D CNN model for CWRU dataset using LinearAsGNN for convolution and pooling.
    """
    def __init__(self, input_length: int = 2048):
        super(CNN1DCWRUAsGNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = Conv1dAsGNN(
            length=input_length,
            kernel_size=64,
            stride=16,
            in_channels=1,
            out_channels=16,
            padding='same',
            bias=True,
        )
        self.relu1 = nn.ReLU()
        out_len1 = math.ceil(input_length / 16)

        # Define the pooling layer
        self.pool1 = MaxPool1dAsGNN(
            length=out_len1,
            kernel_size=2,
            stride=2,
            padding='valid'
        )
        out_len1p = math.floor((out_len1 - 2) / 2) + 1

        # Define the second convolutional layer
        self.conv2 = Conv1dAsGNN(
            length=out_len1p,
            kernel_size=3,
            stride=1,
            in_channels=16,
            out_channels=32,
            padding='same',
            bias=True
        )
        self.relu2 = nn.ReLU()
        out_len2 = math.ceil(out_len1p / 1)

        # Define the second pooling layer
        self.pool2 = MaxPool1dAsGNN(
            length=out_len2,
            kernel_size=2,
            stride=2,
            padding='valid'
        )
        out_len2p = math.floor((out_len2 - 2) / 2) + 1

        # Define the fully connected layers using LinearAsGNN
        self.fc = nn.Sequential(
            LinearAsGNN(32 * out_len2p, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            LinearAsGNN(32, 10)
        )

    def forward(self, x):
        x = self.conv1(x)  
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        B, C, L = x.shape
        x = x.reshape(B, C * L)
        x = self.fc(x)      
        return torch.softmax(x, dim=1)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate models and run summaries
    model1 = CNN1DCWRU().to(device)
    model2 = CNN1DCWRUAsGNN(input_length=2048).to(device)

    # Print summaries
    print("Summary for CNN1DCWRU:")
    summary(model1, (100, 1, 2048))
    print("\n\nSummary for CNN1D_CWRU_GNN:")
    summary(model2, (100, 1, 2048))
