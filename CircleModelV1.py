from torch import nn 

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.four_linear_layers = nn.Sequential(
            nn.Linear(in_features=2,out_features=128),
            # nn.ReLU(),
            # nn.Linear(in_features=128,out_features=256),
            # nn.ReLU(),
            # nn.Linear(in_features=256,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=1),
            )
    def forward(self, x):
        return self.four_linear_layers(x)