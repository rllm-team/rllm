from zipimport import zipimporter
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib import ticker, cm
import os

class LoadData(torch.utils.data.Dataset):

    def __init__(self, X, y, scale_data=True):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class DataNN(torch.nn.Module):
    def __init__(self, channels, acitvation = 'tanh',seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        if acitvation == 'tanh':
            self.acitvation = torch.nn.Tanh()
        elif acitvation == 'relu':
            self.acitvation = torch.nn.ReLU()
        elif acitvation == 'sigmoid':
            self.acitvation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError
            
        hidden_layers = []
        self.num_hidden = len(channels)-1
        for i in range(self.num_hidden):
            hidden_layers.append(torch.nn.Linear(channels[i], channels[i+1], bias=True))
            hidden_layers.append(self.acitvation)
        
        self.hidden_layers = torch.nn.ModuleList(hidden_layers)
    def forward(self, x):
        for i,layer in enumerate(self.hidden_layers):
            x = layer(x)
#             if i < self.num_hidden-1:
#                 x = self.acitvation(x)
        return x

# +
class DataNNTrainer():
    def __init__(self, channels:list, acitvation:str,X_train:np.array, y_train:np.array,X_grid:np.array,batch_size:int,max_epoch:int,lr:float,resolution:tuple,model_save_dir:str):

        assert channels[0] == 2, "only support 2d inputs"
        self.model = DataNN(channels,acitvation)
        print(self.model)
        
        self.traindata = LoadData(X_train, y_train)
        self.trainloader = DataLoader(self.traindata, batch_size=batch_size, shuffle=True)

        self.max_epoch = max_epoch
        # self.loss_function = nn.MSELoss()
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.X_grid = X_grid
        self.channels = channels
        self.acitvation = acitvation
        self.resolution = resolution
        self.X_train = X_train
        self.model_save_dir = model_save_dir
        self.y_train = y_train
   

    def train(self):
        self.model.train()

        for ep in range(self.max_epoch):
            current_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):

                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.long()
                targets = targets.squeeze()

                # Zero the gradients
                self.optimizer.zero_grad()

                # Perform forward pass
                output_logits = self.model(inputs.float())

                # Compute loss
                loss = self.loss_function(output_logits, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                self.optimizer.step()

                # Print statistics
                current_loss += loss.item()

#             print('-------------Epoch:%d------------------'%ep)
#             print("Cross Validation Loss:%.4f"%current_loss)

            if ep%10 == 0:
                print("Cross Validation Loss:%.4f"%current_loss)
                save_path = os.path.join(self.model_save_dir, 'model_at_ep_%d.pth'%ep)
                torch.save({'epoch': ep,
                            'train_loss':current_loss,
                            'model_state_dict': self.model.state_dict()},
                        save_path)
                self.visualize_boudary(ep)
            # TODO: save check point


    def visualize_boudary(self,epoch):

        self.model.eval()
        y_output_logits = self.model(torch.from_numpy(self.X_grid).float())
        y_predicted = torch.argmax(y_output_logits,axis=-1)
        plot_decision_boundry_2d(self.X_grid,y_predicted.detach().numpy(),"Num of Epochs: %d, in a Model: %s, with Activation Func: %s" %(epoch,self.channels,self.acitvation),
                        self.resolution,None,None,os.path.join(self.model_save_dir,'ep_%d.png'%epoch))
        # plot_decision_boundry_2d(self.X_grid,y_predicted.detach().numpy(),"Num of Epochs: %d, in a Model: %s, with Activation Func: %s" %(epoch,self.channels,self.acitvation),
        #                 self.resolution,self.X_train,self.y_train,os.path.join(self.model_save_dir,'ep_%d.png'%epoch))


# -

def plot_decision_boundry_2d(x_grid:np.array,y_grid:np.array,title,resolution,x_scatter,y_scatter,name,discretize=True):
    """
        x_grid: (N*N,2)
        y_grid: (N*N,) - not discretized yet
        x_scatter: (...,2)
        y_scatter: (...,2)
    """
    z = y_grid.reshape((resolution[0],resolution[1]))
    if discretize:
        z = np.where(z>0.5,1,0)
        # z = np.where(z>0,1,-1)
    x_grid_1 = x_grid[:,0].reshape((resolution[0],resolution[1]))
    x_grid_2 = x_grid[:,1].reshape((resolution[0],resolution[1]))
    ax = plt.figure()
    ax.set_facecolor('white')
    plt.contourf(x_grid_1, x_grid_2, z,cmap ="ocean")
    if x_scatter is not None and y_scatter is not None:
        plt.scatter(x_scatter[:,0],x_scatter[:,1],c=y_scatter,cmap ="ocean",edgecolors='black')
      
    plt.title(title)
    plt.savefig(name,bbox_inches='tight',dpi=300)

