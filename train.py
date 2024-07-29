# Import libraries
import os, argparse, torch, importlib, cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from CNNs import SimpleCNN_p32 # Need to be Fixed 



parser = argparse.ArgumentParser()

parser.add_argument('--patchSize',
                    required=True,
                    type=int)

parser.add_argument('--TrainingPercentage',
                    default=0.8,
                    type=float)

parser.add_argument('--TestingPercentage',
                    default=0.2,
                    type=float)

parser.add_argument('--CNN',
                    required=True)

parser.add_argument('--epochs',
                    default=200, 
                    type=int, )

parser.add_argument('--device',
                    default="mps", 
                    type=str)

args = parser.parse_args()

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

class algorithm:
    """
    The algorithm class has its own custom dataset 
    1. Apply transformation ---> dataloader
    2. Model ---> CNNs
    3. Training model ---> epoch, loss fuction, and optimizer
    4. Evaluating stage ---> Confusion Matrix 
    """
    def getModel(cnn_name):
        """
        This functions will get the model name, iterate through
        all the potential models in dictionary and return it
        """
        cnn_models = {
        "SimpleCNN_p32": SimpleCNN_p32.SimpleCNN_p32()}
        
        if cnn_name in cnn_models:
            cnnModel = cnn_models[cnn_name]
        else:
            raise ValueError(f"Unsupported CNN Model: {cnn_name}")
        return cnnModel
    
    # Hyper Parameters Definitions from script 
    patchSize = args.patchSize
    train_size = args.TrainingPercentage
    test_size = args.TestingPercentage
    cnn_name = args.CNN
    cnnModel = getModel(cnn_name)
    num_epochs = args.epochs
    device = args.device

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
    datadir = f'Preprocessing/output_patches/patch_size/{patchSize}X{patchSize}'
    dataset = SweatPoresDataset(img_dir = datadir, transforms = trans)
    
    # Split the data -- Train Validate Test
    train_data, test_data = train_test_split(dataset, test_size=test_size, train_size=train_size)
    
    # Testing out the dataset 
    print(f"Total data: {len(dataset)}")
    print(f"Training images dataset: {len(train_data)}")
    print(f"Testing images dataset: {len(test_data)}")
    
     
    # Pass data into dataloader 
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers = 0)
    test_loader = DataLoader(test_data, batch_size = 8, shuffle = True, num_workers = 0)
    
    # Testing out the dataloader
    X, Y, Z = next(iter(train_loader))
    print(f"Data: {X.shape}")
    print(f"Lable: {Y[0]}")
    print(f"Image names: {Z[0]}")
    
    # Define the loss function and optimizer 
    loss_fn = nn.CrossEntropyLoss(weight = torch.Tensor([1.0,4.0]).to('mps'))
    optimizer = optim.SGD(cnnModel.parameters(), lr=0.001, momentum=0.9)
    
    def trainModel(num_epochs, train_loader, device, cnnModel, optimizer, loss_fn):
        cnnModel.to(device)
        train_loss = []
        for epoch in range(num_epochs):
            # set the model to train
            cnnModel.train()
            running_loss = 0.0
            for images, labels, _ in train_loader:
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)
                
                # foward Propagation 
                optimizer.zero_grad() # Reset the gradient from the previous calculation
                outputs = cnnModel(images)
                loss = loss_fn(outputs, labels)

                # Back Propagation ---> Update the model weights/bias in each steps
                loss.backward()
                optimizer.step()

                # Keep track of the running loss
                running_loss += loss.item() * labels.size(0)
        
            train_loss = running_loss / len(train_loader.dataset)
            print(f"\rEpoch {epoch+1}/{num_epochs} - Train loss: {train_loss}", end='', flush=True)

        print()
        return cnnModel
    
    trainedModel = trainModel(num_epochs, train_loader, device, cnnModel, optimizer, loss_fn)
    
    # Save the model
    modelName = f'{cnn_name}_e{num_epochs}'
    torch.save(trainedModel.state_dict(), f'models/{modelName}.model')
    
    def evaluateModel(test_loader, device, model):
        # set the model to evaluation mode 
        model.eval()
        
        fp_names = [] # Image names for false.
        fn_names = [] # Images where we missed predicting the pore.
        tp_names = []
        tn_names = []
        confusionMatric = { # Dict of confusion matix
            "TP: ":0,
            "TN: ":0,
            "FP: ":0,
            "FN: ":0}

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # disable gradient calculation
        with torch.no_grad():
            for images, labels, names in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                prediction = model(images)

                # Apply softmax to convert the raw logits into probabilities
                probabilities = F.softmax(prediction, dim=1)

                # Finds the highest probabilities for each image
                _, predicted_classes = torch.max(probabilities, 1)

                for i in range(len(labels)):
                    if predicted_classes[i].item() == labels[i].item() == 1:
                        tp_names.append(names[i])
                        TP += 1
                    elif predicted_classes[i].item() == labels[i].item() == 0:
                        tn_names.append(names[i])
                        TN += 1
                    elif predicted_classes[i].item() != labels[i].item() and predicted_classes[i].item() == 1:
                        fp_names.append(names[i]) # Keep track of the fp
                        FP += 1
                    else:
                        fn_names.append(names[i]) # Keep track of the fn
                        FN += 1
                        
        confusionMatric["TP: "] = TP
        confusionMatric["FP: "] = FP
        confusionMatric["TN: "] = TN
        confusionMatric["FN: "] = FN
                
        return fp_names, fn_names, tn_names, tp_names, [TP, TN, FP, FN]
    
    fp, fn, tn, tp, results= evaluateModel(test_loader, device, trainedModel)
    
    def ConfusionMatrix(results, modelName):
        TP, TN, FP, FN = results
        
        # Create the full file path
        file_path = os.path.join('experiments/', modelName+'.txt')

        with open(file_path, 'w') as f:
            f.write("--- Confusion Matrix ---\n")
            f.write(f"Model: {modelName}\n")
            f.write(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}\n")
            
            if TP + FP + FN + TN > 0:
                accuracy = (TP + TN) / (TP + FP + FN + TN)
                f.write(f"Accuracy: {accuracy:.4f}\n")
            else:
                f.write("Accuracy: N/A\n")
            
            if TP + FP > 0:
                precision = TP / (TP + FP)
                f.write(f"Precision: {precision:.4f}\n")
            else:
                f.write("Precision: N/A (no positive predictions)\n")

            if TP + FN > 0:
                recall = TP / (TP + FN)
                f.write(f"Recall: {recall:.4f}\n")
            else:
                f.write("Recall: N/A (no actual positive samples)\n")

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                f.write(f"F-score: {f1:.4f}\n")
            else:
                f.write("F-score: N/A (precision and recall are both zero)\n")
        return 
    
    ConfusionMatrix(results, modelName)

if __name__ == "__main__":        
    algorithm()