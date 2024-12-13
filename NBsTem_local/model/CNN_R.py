import torch
import torch.nn as nn
import math

class Network(nn.Module):    
    def __init__(self, input_channel=512, num_classes=1):    
        super(Network, self).__init__()
        self.cnn1 = nn.Sequential(    
            nn.Conv1d(input_channel, out_channels=512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            nn.ReLU(),   
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(128),    
            nn.ReLU(),    
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )
        self.fc2 = nn.Sequential(    
            nn.Linear(6400, 256),    
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)   
        )
    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)
        x = self.cnn1(x)
        x = x.reshape(-1, 6400)
        out = self.fc2(x)
        out = out.view(-1)
        return out
