from torch import nn

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=2, out_features=128) # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(in_features=128, out_features=5) # takes in 5 features, produces 1 feature (y)
        self.layer_3 = nn.Linear(in_features=5, out_features=1)
        self.Relu = nn.ReLU()
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # Return the output of layer_2, a single feature, the same shape as y
        return self.layer_3(self.Relu(self.layer_2(self.Relu(self.layer_1(x))))) # computation goes through layer_1 first then the output of layer_1 goes through layer_2
