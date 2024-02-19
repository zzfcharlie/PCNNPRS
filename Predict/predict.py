import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


class Net(nn.Module):
    def __init__(self, alpha):
        super(Net, self).__init__()
        self.l1 = nn.ModuleList([nn.Linear(1400, 1) for _ in range(22)])
        self.l2 = nn.Linear(22 * 1, 1000)
        self.l3 = nn.Linear(1000, 100)
        self.l4 = nn.Linear(100, 50)
        self.l5 = nn.Linear(50, 10)
        self.l6 = nn.Linear(10, 1)
        self.output = [0] * 22
        self.alpha = alpha

    def forward(self, x):
        for i in range(22):
            x_1 = x[:, i * 1400:(i + 1) * 1400]
            self.output[i] = self.l1[i](x_1)
        x = torch.cat(self.output, dim=1)
        x = F.leaky_relu(self.l2(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l3(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l4(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l5(x), negative_slope=self.alpha)
        x = self.l6(x)
        return x


class ReduceNet(nn.Module):
    def __init__(self, alpha):
        super(ReduceNet, self).__init__()
        self.l2 = nn.Linear(22, 1000)
        self.l3 = nn.Linear(1000, 100)
        self.l4 = nn.Linear(100, 50)
        self.l5 = nn.Linear(50, 10)
        self.l6 = nn.Linear(10, 1)
        self.output = [0] * 22
        self.alpha = alpha

    def forward(self, x):
        x = F.leaky_relu(self.l2(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l3(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l4(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l5(x), negative_slope=self.alpha)
        x = self.l6(x)
        return x


def predict(material_dir, result_dir):
    reduce_model_path = f'{material_dir}/reduce_model.pkl'
    scalery = torch.load(f'{material_dir}/scalery.pkl')
    hyperparameters = torch.load(
        f'{material_dir}/best_hyperparams.pkl')
    learning_rate = hyperparameters['learning_rate']
    batch_size = hyperparameters['batch_size']
    weight_decay = hyperparameters['weight_decay']
    alpha = hyperparameters['alpha']
    reduce_model = torch.load(reduce_model_path)
    x_test = np.array(pd.read_csv(f'{material_dir}/prs22.csv', header=None))
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    with torch.no_grad():
        pred = scalery.transform(reduce_model(x_test_tensor))
    np.savetxt(f'{result_dir}/pred.txt', pred)


predict(material_dir, result_dir)
