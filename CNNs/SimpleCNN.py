import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        # Call the initializer for the parent class
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)     # in_channnels = 1 / out_channels = 6 / kernel size = 3
        self.conv2 = nn.Conv2d(6,16,3)    # 6 ---> 16
        self.maxpool = nn.MaxPool2d(2, 2) # kernel size = 2 / strides = 2
        self.relu = nn.ReLU()
        # flatten out the channels(features) with width and height ---> C * W * H
        self.flatten = nn.Flatten(start_dim=1)  # it will return a 2D tensor, preserve the batch dimension and flatten everything else
        self.fc1 = nn.Linear(16 * 36, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 40)
        self.fc4 = nn.Linear(40,2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Flatten the tensor, preserve the batch dimension and flatten everything else
        x = self.flatten(x) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x