import data_manipulation
import torch.nn as nn
import numpy as np

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)


cloud_dict = data_manipulation.load_occupancy_grids()
for v in cloud_dict.values():
    break
model = TestModel()
model.forward(v)
