import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import time
import random
device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self, alpha):
        super(Net, self).__init__()
        self.l1 = nn.ModuleList([nn.Linear(1400, 1) for _ in range(22)])
        self.l2 = nn.Linear(22 * 1, 1000)
        self.l3 = nn.Linear(1000, 100)
        self.l4 = nn.Linear(100, 50)
        self.l5 = nn.Linear(50, 10)
        self.l6 = nn.Linear(10,1)
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
    def __init__(self, alpha, pretrained_model):
        super(ReduceNet, self).__init__()
        self.l2 = pretrained_model.l2
        self.l3 = pretrained_model.l3
        self.l4 = pretrained_model.l4
        self.l5 = pretrained_model.l5
        self.l6 = pretrained_model.l6
        self.output = [0] * 22
        self.alpha = alpha
        
    def forward(self, x):
        x = F.leaky_relu(self.l2(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l3(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l4(x), negative_slope=self.alpha)
        x = F.leaky_relu(self.l5(x), negative_slope=self.alpha)
        x = self.l6(x)
        return x
    

def get_data(multi_train,y_train):
    print('loading_data ...')
    x = multi_train
    y = y_train
    print('loading_complete!')
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2)
    scalerx = StandardScaler()
    scalerx.fit(x_train)    
    scalery = StandardScaler()
    scalery.fit(y_train)
    x_train = scalerx.transform(x_train)
    x_val = scalerx.transform(x_val)    
    y_train = scalery.transform(y_train)
    y_val = scalery.transform(y_val)
    return x_train,x_val,y_train,y_val,scalerx,scalery



def get_dataloader(x_train,x_val,y_train,y_val,batch_size):
    print('get_dataloader...')
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    torch_train_data = TensorDataset(x_train_tensor,y_train_tensor)
    torch_val_data = TensorDataset(x_val_tensor,y_val_tensor)
    train_dataloader = DataLoader(torch_train_data, batch_size=(batch_size), shuffle=True)
    val_dataloader = DataLoader(torch_val_data, batch_size=(y_val.shape[0]), shuffle=True)
    print('complete!')
    return train_dataloader, val_dataloader




def train(alpha,learning_rate, batch_size, weight_decay,patience, x_train, x_val, y_train, y_val):
    epochs = 200
    alpha = alpha
    lr = learning_rate
    model = Net(alpha).to(device)
    train_dataloader, val_dataloader = get_dataloader(x_train,x_val,y_train,y_val,batch_size)
    weight = weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=400, gamma=0.7)
    count = 0
    val_loss_min = float('inf')
    pre_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        loss_fn = nn.MSELoss()
        total_train_loss = 0
        total_train_r2 = 0
        for data in train_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = loss_fn(outputs,y)
            optimizer.zero_grad()
            with torch.no_grad():
                total_train_loss = total_train_loss + loss.item()
                total_train_r2 = total_train_r2 + r2_score((y.cpu()).detach().numpy(), (outputs.cpu()).detach().numpy())
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_lr = (scheduler.get_last_lr())
        train_loss = (total_train_loss/len(train_dataloader))
        train_R2 = (total_train_r2/len(train_dataloader))
        total_val_loss = 0
        total_val_r2=0

        if ((epoch + 1) % 3 == 0):
            model.eval()
            with torch.no_grad():
                for data in val_dataloader:
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    total_val_loss = total_val_loss + loss.item()
                    total_val_r2 = total_val_r2 + r2_score((y.cpu()).detach().numpy(), (outputs.cpu()).detach().numpy())
                val_loss = (total_val_loss / len(val_dataloader))
                val_R2 = (total_val_r2 / len(val_dataloader))
                if (val_loss < val_loss_min):
                    val_loss_min = val_loss
                    count = 0
                else:
                    count += 1
                if abs(val_loss - pre_val_loss) < 1e-5:
                    print('Convergence reached.')
                    break
                pre_val_loss = val_loss
            print(
                f'Epoch: {epoch},Train Loss: {train_loss:.4f},Val Loss: {val_loss:.4f} Train R2: {train_R2:.4f}, Val R2: {val_R2:.4f}')
        if (count == patience):
            break
    return model, val_R2, val_loss


param_grid = {
    'learning_rate': [0.00002],
    'batch_size': [128,256,512],
    'weight_decay': list(np.logspace(np.log10(1e-8), np.log10(1e-5), base=10, num=100)),
    'alpha': list(np.linspace(0.01, 1.0, num=100)),
    'patience': list(range(3, 5))
}




torch.manual_seed(int(seed))
np.random.seed(int(seed))
random.seed(int(seed))
torch.set_num_threads(int(Ncores))
MAX_EVALS = int(max_evals)



def Random_Search(param_grid,MAX_EVALS):
    x_train,x_val,y_train,y_val,scalerx,scalery = get_data(multi_train,y_pheno)
    torch.save(scalerx,f'{out_dir}/scalerx.pkl')
    torch.save(scalery,f'{out_dir}/scalery.pkl')
    start_time_all = time.time()
    best_score = 0
    best_hyperparams = {}
    for i in range(MAX_EVALS):
        start_time = time.time()
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        learning_rate = hyperparameters['learning_rate']
        batch_size = hyperparameters['batch_size']
        weight_decay = hyperparameters['weight_decay']
        alpha = hyperparameters['alpha']
        patience = hyperparameters['patience']
        print(f'Round {i+1} of Random Search begins，hyperparameters list:{hyperparameters}')
        model, val_R2, val_loss = train(alpha,learning_rate, batch_size, weight_decay, patience, x_train,x_val,y_train,y_val)
        print(f'Round {i+1} of Random Search completed, elapsed time:{(time.time()-start_time)/60:.4f}min.')
        score = val_R2
        if score > best_score:
            best_hyperparams = hyperparameters
            best_model = model
            best_score = score
            merged_weights = torch.Tensor()
            merged_bias = torch.Tensor()
            for linear_layer in best_model.l1:
                merged_weights = torch.cat((merged_weights, linear_layer.weight.view(-1)))
                merged_bias = torch.cat((merged_bias, linear_layer.bias.view(1)))
            prs_coeff = merged_weights.detach().numpy()/np.sqrt(np.where(scalerx.var_==0,1,scalerx.var_))
            std_bias = -(prs_coeff*scalerx.mean_).reshape(22,1400).sum(axis=1)
            prs_bias = std_bias + merged_bias.detach().numpy()
            reduce_model = ReduceNet(best_hyperparams['alpha'], best_model)
            np.savetxt(f"{out_dir}/prs_coeff.txt",prs_coeff)
            np.savetxt(f"{out_dir}/prs_bias.txt",prs_bias)
            torch.save(best_hyperparams,f"{out_dir}/best_hyperparams.pkl")
            torch.save(best_model, f"{out_dir}/best_model.pkl")
            torch.save(reduce_model, f"{out_dir}/reduce_model.pkl")  
    end_time_all = time.time()
    print(f'Total elapsed time:{(end_time_all-start_time_all)/60:.4f} min.')

    


if __name__ == '__main__':
    Random_Search(param_grid,MAX_EVALS)

    
    


