from collections import OrderedDict

import torch.nn as nn


class ModulationCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1",nn.Conv1d(
                in_channels = 2, out_channels = 32,
                kernel_size = 5, stride = 1, padding =2
            )),
            ("bn1", nn.BatchNorm1d(32)),
            ("relu1", nn.ReLU()),
            ("pool",nn.MaxPool1d(kernel_size = 2)),
            
            ("conv2", nn.Conv1d(
                in_channels = 32, out_channels = 64,
                kernel_size = 3, stride = 1,  padding = 1
            )),
            ("bn2",nn.BatchNorm1d(64)),
            ("relu2",nn.ReLU()),
            ("avg_pool",nn.AdaptiveAvgPool1d(1)),
        ]))

        self.classifier = nn.Linear(64,4)

    def forward(self,x):
        x = self.features(x)    #(batch, 64, 1)
        x = x.squeeze(-1)         #(batch, 64)
        return self.classifier(x)   #(batch, 4)
    