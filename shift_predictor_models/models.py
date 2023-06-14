import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_dim,class_dim, inner_dim, output_dim):
        super(MLP, self).__init__()
        self.fc_direction = nn.Sequential(
            nn.Linear(input_dim+class_dim,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim,output_dim),
        )
        
    def forward(self, x,c):
        x_c = torch.cat((x,c),1)
        dir_ = self.fc_direction(x_c)
        return dir_

class MLPDropout(nn.Module):
    def __init__(self,input_dim,class_dim, inner_dim, output_dim):
        super(MLPDropout, self).__init__()
        self.fc_direction = nn.Sequential(
            nn.Linear(input_dim+class_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(inner_dim,output_dim),
        )
        
    def forward(self, x,c):
        x_c = torch.cat((x,c),1)
        dir_ = self.fc_direction(x_c)
        return dir_
    
class MLPThreeLayer(nn.Module):
    def __init__(self, input_dim, class_dim, inner_dim, output_dim):
        super(MLPThreeLayer, self).__init__()
        self.fc_direction = nn.Sequential(
            nn.Linear(input_dim+class_dim,512),
            nn.ReLU(),
            nn.Linear(512,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim,output_dim),
        )
        
    def forward(self, x,c):
        x_c = torch.cat((x,c),1)
        dir_ = self.fc_direction(x_c)
        return dir_
    
class MLPThreeLayerDropout(nn.Module):
    def __init__(self, input_dim, class_dim, inner_dim, output_dim):
        super(MLPThreeLayerDropout, self).__init__()
        self.fc_direction = nn.Sequential(
            nn.Linear(input_dim+class_dim,512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(inner_dim,inner_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(inner_dim,output_dim),
        )
        
    def forward(self, x,c):
        x_c = torch.cat((x,c),1)
        dir_ = self.fc_direction(x_c)
        return dir_