import torch
import torch.nn as nn

class SimpleCNN_p17(nn.Module):
    def __init__(self):
        # Call the initializer for the parent class
        super(SimpleCNN_p17, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)     # in_channnels = 1 / out_channels = 6 / kernel size = 3
        self.conv2 = nn.Conv2d(6,16,3)    # 6 ---> 16
        self.maxpool = nn.MaxPool2d(2, 2) # kernel size = 2 / strides = 2
        self.relu = nn.ReLU()
        # flatten out the channels(features) with width and height ---> C * W * H
        self.flatten = nn.Flatten(start_dim=1)  # it will return a 2D tensor, preserve the batch dimension and flatten everything else
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8,2)
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        #print(f"After conv1: {x.shape}")
        
        x = self.relu(x)
        #print(f"After first ReLU: {x.shape}")
        
        x = self.maxpool(x)
        #print(f"After first maxpool: {x.shape}")
        
        x = self.conv2(x)
        #print(f"After conv2: {x.shape}")
        
        x = self.relu(x)
        #print(f"After second ReLU: {x.shape}")
        
        x = self.maxpool(x)
        #print(f"After second maxpool: {x.shape}")
        
        x = self.flatten(x)
        #print(f"After flatten: {x.shape}")
        
        x = self.fc1(x)
        #print(f"After fc1: {x.shape}")
        
        x = self.fc2(x)
        #print(f"After fc2: {x.shape}")
        
        x = self.fc3(x)
        #print(f"After fc3: {x.shape}")
        
        x = self.fc4(x)
        #print(f"Output shape: {x.shape}")
        
        return x
    
    

# # Create an instance of the model
# model = SimpleCNN_p17()

# # Create a dummy input tensor with batch size 8
# dummy_input = torch.randn(8, 1, 17, 17)

# # Pass the dummy input through the model
# output = model(dummy_input)

# print(f"\nFinal output shape: {output.shape}")