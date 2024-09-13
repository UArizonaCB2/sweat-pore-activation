from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import os, argparse, torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser.add_argument('--patchSize',
                    type=int)

parser.add_argument('--processedImg',
                    type=str)

parser.add_argument('--TrainingPercentage',
                    default=0.8,
                    type=float)

parser.add_argument('--TestingPercentage',
                    default=0.2,
                    type=float)

args = parser.parse_args()

class SweatPoresDataset(Dataset):
    def __init__(self, img_dir, transforms = None):
        """
        It is run once when instantiating the Dataset object. 
        """
        self.img_dir = img_dir
        self.transform = transforms
        self.img_files = os.listdir(img_dir) # list of image files

    def __len__(self):
        """
        It returns the number of samples in our dataset
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        It loads and returns a smaple from the dataset at the given index
        """
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB') # Retrieve the image data 
        label = int(self.img_files[idx].split('_')[-1].split('.')[0]) # Etrack the label from the image
        img_name = self.img_files[idx] 

        if self.transform: 
            image = self.transform(image)

        return image, label, img_name

if __name__ == "__main__":
    print("-- Dataset Info --")
    patchSize = args.patchSize
    processedImg = args.processedImg
    train_size = args.TrainingPercentage
    test_size = args.TestingPercentage
    
    # Create Transformation
    """
    Converting Images to Tensor and normalize the pixel values
    """
    trans = transforms.Compose([
        transforms.Grayscale(),  # Convert RGB to grayscale 3 ---> 1 
        transforms.ToTensor(), # Convert data to Tensor for dataloader
        transforms.Normalize(mean=0.5, std=0.5) # scale the pixel values of the image between -1 and 1
    ])
    
    # Apply transformation on the dataset 
    # datadir = f'Preprocessing/output_patches/patch_size/{patchSize}X{patchSize}'
    datadir = f'Preprocessing/testingModel_output_patches/{processedImg}/patch_size/{patchSize}X{patchSize}'
    dataset = SweatPoresDataset(img_dir = datadir, transforms = trans)
    
    # Split the data indices -- Train Validate Test
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, train_size=train_size)
    
    torch.save(dataset, f'Preprocessing/dataset/{patchSize}X{patchSize}/dataset.pt')
    torch.save(train_indices, f'Preprocessing/dataset/{patchSize}X{patchSize}/train_indices.pt')
    torch.save(test_indices, f'Preprocessing/dataset/{patchSize}X{patchSize}/test_indices.pt')
    
    # Testing out the dataset 
    print(f"Preprocessing on the image: {processedImg}")
    print(f"Total data: {len(dataset)}")
    print(f"Training dataset: {len(train_indices)}")
    print(f"Testing dataset: {len(test_indices)}")