import late_fusion
import early_fusion
import torch.optim as optim
import data_manipulation


def train_model(model):
    optimizer = optim.Adam(parameters=model.parameters(), lr=0.0001)
