import torch.nn as nn

class ModulationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2,32,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )   #卷积等网络层
        self.classifier = nn.Linear(64, 4)

    def forward(self,x):
        x = self.features(x)    #(batch, 64, 1)
        x = x.squeeze(-1)  #(batch, 64)
        return self.classifier(x)   #(batch,4),四类的原始分数
