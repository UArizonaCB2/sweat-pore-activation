import torch.nn as nn

class CNN4Layers_p32(nn.Module):
    def __init__(self):
        # Call the initializer for the parent class
        super(CNN4Layers_p32, self).__init__()
        
        self.conv1 = nn.Conv2d(1,8,3)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,64,3)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2) # kernel size = 2 / strides = 2
        self.relu = nn.ReLU()
        # flatten out the channels(features) with width and height ---> C * W * H
        self.flatten = nn.Flatten(start_dim=1)  # it will return a 2D tensor, preserve the batch dimension and flatten everything else
        self.fc1 = nn.Linear(256,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,2)
        
    def forward(self,x):
        """
        Notice: 
            Change Tensor Size :
                Maxpooling
            Chnage Tensor Value(feature map):
                BatchNormalization
                ReLU
        """
        x = self.maxpool(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchnorm2(self.conv2(x))))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.relu(self.batchnorm4(self.conv4(x)))
        # Flatten the tensor, preserve the batch dimension and flatten everything else
        x = self.flatten(x) 
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
    
    
        # print(f"Input shape: {x.shape}")
            
        # x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        # x = self.batchnorm1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # print(f"After maxpool1: {x.shape}")
        
        # x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        # x = self.batchnorm2(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # print(f"After maxpool2: {x.shape}")
        
        # x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        # x = self.batchnorm3(x)
        # x = self.relu(x)
        
        # x = self.conv4(x)
        # print(f"After conv4: {x.shape}")
        # x = self.batchnorm4(x)
        # x = self.relu(x)
        
        # x = self.flatten(x)
        # print(f"After flatten: {x.shape}")
        
        # return x
        
    






 
# import torch
# # Create a sample tensor with shape [8, 1, 32, 32]
# sample = torch.randn(8, 1, 32, 32)
# print(sample.shape)

# print("this is VGG16 model")

# model = CNN4Layers_p32()
# output = model(sample)

# print(output)

