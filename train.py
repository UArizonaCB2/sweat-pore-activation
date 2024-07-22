# Import libraries
import os, argparse
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

parser.add_argument('--patchSize',
                    required=True,
                    type=int)

args = parser.parse_args()


class algorithm:
    """
    The algorithm class has its own custom dataset 
    1. Apply transformation ---> dataloader
    2. Model ---> CNNs
    3. Training model ---> epoch, loss fuction, and optimizer
    4. Evaluating stage ---> Confusion Matrix 
    """
    # Custom Dataset
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
    datadir = f'Preprocessing/output_patches/batch_sizes/{args.patchSize}X{args.patchSize}'
    dataset = SweatPoresDataset(img_dir = datadir, transforms = trans)
    
    # Split the data
    # Training: 80%
    # Testing: 20%
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    print(f"Total data: {len(dataset)}")
    print(f"Training images dataset: {len(train_data)}")
    print(f"Testing images dataset: {len(test_data)}")

if __name__ == "__main__":
    algorithm()