import torch.nn as nn 


class EarlyFusion(nn.Module):

    def __init__(self):
        super(EarlyFusion, self).__init__()
        self.configs = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256]
        self.features = self.generate_features()
    
    def generate_features(self):
        layers = []
        in_channels = 29
        layers += [nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5), nn.ReLU(inplace=True)]
        for v in self.configs:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.features(x)