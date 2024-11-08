import os, cv2, torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from CNNs import CNN4Layers_p32
import torch.nn.functional as F

class PatchDataset(Dataset):
    def __init__(self, patchList, transform=None):
        self.patchList = patchList
        self.transform = transform
        
    def __len__(self):
        return len(self.patchList)
    
    def __getitem__(self, idx):
        patch = self.patchList[idx]
        # Convert NumPy array to PIL Image
        PIL_image = Image.fromarray(patch)
        if self.transform:
            patch = self.transform(PIL_image)
        return patch

class centeredPixel:
    """
    This class will centerlized each pixel in the image
    and create patches with a center of each pixel
    """
    def __init__(self):
        self.cnn_name = "CNN4Layers_p32_e20_135bmps.model"
        self.imgNumber = "2"
        self.imgDir = f"./Preprocessing/input_images/testingModel/{self.imgNumber}bmp/raw/"
        padded_img = self.addPaddings(self.imgDir, self.imgNumber)
        patchList = self.createPatches(padded_img)
        dataloader = self.createDataLoader(patchList)
        model = self.getCNNmodel(self.cnn_name)
        self.prediction(model, dataloader)
    
    def addPaddings(self, imgDir, imgName):
        """
        Add paddings around the original image
        with the size of patch // 2
        """
        # Construct the full path to the image
        image_path = os.path.join(imgDir, f"{imgName}.bmp")
        image = cv2.imread(image_path)
        
        # Get image dimensions
        height, width, channels = image.shape
        print(f'Img shape - height:{height}, width:{width}, channels:{channels}')
        
        # Define padding size
        paddings = 32 // 2
        # Add paddings
        padded_image = cv2.copyMakeBorder(
            image,
            top=paddings,
            bottom=paddings,
            left=paddings,
            right=paddings,
            borderType=cv2.BORDER_CONSTANT,
            value=[255 , 255, 255]  # white color for padding
        )
        
        # Get padded image dimensions
        padded_height, padded_width, padded_channels = padded_image.shape
        print(f'Padded img shape - height:{padded_height}, width:{padded_width}, channels:{padded_channels}')
        
        return padded_image
    
    def createPatches(self, padded_img):
        
        height, width, channels = padded_img.shape
        
        patchLst = []
        patchSize = 32
        # Iterate over the image to create patches, moving one pixel at a time
        for y in range(height - patchSize + 1):
            for x in range(width - patchSize + 1):
                patch = padded_img[y:y + patchSize, x:x + patchSize]
                patchLst.append(patch)
        
        return patchLst
        
    def createDataLoader(self, patchList):
        """
        This function create data loaders for dataset 
        """
        trans = transforms.Compose([
            transforms.Grayscale(),  # Convert RGB to grayscale 3 ---> 1 
            transforms.ToTensor(), # Convert data to Tensor for dataloader
            transforms.Normalize(mean=0.5, std=0.5) # scale the pixel values of the image between -1 and 1
        ])
        
        # Create a dataset from patchList and apply transformation
        dataset = PatchDataset(patchList, transform=trans)
        
        # Create a dataLoader with batch_size = 8
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        return dataloader
    
    def getCNNmodel(self, cnn_name):
        # Load the state dict
        model_path = f'./models/{cnn_name}'
        state_dict = torch.load(model_path)

        model = CNN4Layers_p32.CNN4Layers_p32()
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)

        return model
              
    def prediction(self, model, dataloader):
        # Set up the device
        device = 'mps'
        model = model.to(device)

        # disable gradient calculation
        with torch.no_grad():
            model.eval()
            for images in dataloader:
                
                images = images.to(device)
                
                prediction = model(images)
                
                # Apply softmax to convert the raw logits into probabilities
                probabilities = F.softmax(prediction, dim=1)
                
                print(f"Probabilities {probabilities}")
        return 
        
if __name__ == "__main__":
    centeredPixel()