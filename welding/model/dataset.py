import torch
class weld_dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self,index):
        x_item = torch.tensor(self.x[index], dtype=torch.float32)
        y_item = torch.tensor(self.y[index], dtype=torch.float32)
        return x_item,y_item