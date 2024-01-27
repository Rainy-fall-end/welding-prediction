from configparser import ConfigParser

import sys
# Hyper-parameters
config = ConfigParser()
config.read(sys.path[0].replace('\\','/') + "/config.ini")
hyper = config['Hyper']
data = config['data']

target_path="./"
sys.path.append(config['path']['project_path'])

import torch
import sys
from model.utils import load_data
from model.dataset import weld_dataset
from neural_network import NeuralNet
import torch.nn as nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
welding_path = data['welding_path']
para_path = data['para_path']
x_data,y_data,grid = load_data(welding_path,para_path)
train_dataset = weld_dataset(x_data,y_data)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=int(hyper['batch_size']), 
                                           shuffle=True)

input_size = x_data.shape[1]
out_size = y_data.shape[1]
model = NeuralNet(input_size, int(hyper['hidden_size']), out_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(hyper['learning_rate']))
  
# Train the model
total_step = len(train_loader)
for epoch in range(int(hyper['num_epochs'])):
    for i, (x_, y_) in enumerate(train_loader):  
        # Move tensors to the configured device
        x_ = x_.to(device)
        y_ = y_.to(device)
        
        # Forward pass
        outputs = model(x_)
        loss = criterion(outputs, y_)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, hyper['num_epochs'], i+1, total_step, loss.item()))
res = model(torch.tensor(x_data[8], dtype=torch.float32))

from model.mesh import Mesh
from plots.plot_mesh import plot_func
with torch.no_grad():
    mesh = Mesh(grid,res.numpy())
plot_func.plot_mesh(mesh,"D:\\User\\Jinke\\welding\\predict\\full_connected\\image\\mesh.jpg")
