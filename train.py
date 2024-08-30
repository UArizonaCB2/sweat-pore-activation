# Import libraries
import os, argparse, torch, importlib, cv2
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from CNNs import SimpleCNN_p32, SimpleCNN_p17, CNN4Layers_p32 # Need to be Fixed 



parser = argparse.ArgumentParser()

parser.add_argument('--batchSize',
                    required=True,
                    type=int)

parser.add_argument('--CNN',
                    required=True)

parser.add_argument('--epochs',
                    default=200, 
                    type=int)

parser.add_argument('--device',
                    default="mps", 
                    type=str)

parser.add_argument('--tag',
                    default=None)

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
        "SimpleCNN_p32": SimpleCNN_p32.SimpleCNN_p32(),
        "SimpleCNN_p17": SimpleCNN_p17.SimpleCNN_p17(),
        "CNN4Layers_p32": CNN4Layers_p32.CNN4Layers_p32()}
        
        if cnn_name in cnn_models:
            cnnModel = cnn_models[cnn_name]
        else:
            raise ValueError(f"Unsupported CNN Model: {cnn_name}")
        return cnnModel
    
    # Hyper Parameters Definitions from script 
    batchSize = args.batchSize
    cnn_name = args.CNN
    patchSize = int(cnn_name.split("_p")[1])
    cnnModel = getModel(cnn_name)
    num_epochs = args.epochs
    device = args.device
    
    # Load the testing dataset
    train_data = torch.load(f'Preprocessing/dataset/{patchSize}X{patchSize}/train_indices.pt')
    print(f'Total training data: {len(train_data)}')
    print()
    
    def ratio_of_labels(dataloader):
        """
        Param: dataloader
        Return: Tensor that contains the ratio for labels
        This function counts the labels from the dataloaders
        """
        label_0 = 0 # Does not have sweat pores
        label_1 = 0 # Has sweat pores
        
        for data in dataloader:
            _, labels, _ = data  # labels is a tensor with the length of batch size 
            
            for label in labels:
                if label == 0:
                    label_0+=1
                elif label == 1:
                    label_1 += 1
        
        rate = (label_0/label_1)
        ratio = torch.Tensor([1.0, rate])
        return ratio
    
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
    
    def validationModel(test_loader, device, model):
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
    
    def stratified_KFold_loader(dataset, batchSize=batchSize):
        """
        This function takes a dataset and batchSize and returns fold and dataloaders
        Dataloaders: train_loader, test_loader
                    each loader should have a similar distributed labels
        """
        # Extract tensors and label for stratification into a list
        tensors = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]
        
        skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
        
        # iterate through each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(tensors, labels)):
            # when we use the dataloader, it randomly samples from these indices.
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            # pass the train and test dataloader from the specify samplers
            train_loader = DataLoader(dataset, batch_size=batchSize, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batchSize, sampler=val_sampler)
            
            yield fold, train_loader, val_loader
            
    def analyze_dataloader(dataloader):
        """
        This functions takes arguments: fold and dataloader
        """
        all_labels = []
        label0s = 0
        label1s = 0
        total_samples = 0
        
        for batch in dataloader:
            features, labels, names = batch
            
            # Convert labels to a list and extend all_labels
            all_labels.extend(labels.tolist())

            total_samples+=len(labels)

        for label in all_labels:
            if label == 0:
                label0s +=1
            elif label == 1:
                label1s +=1  
                
        print(f'(label 0: {label0s} , label 1: {label1s})')      
        return 
    
    def PrintConfusionMatrix(results, modelName):
        TP, TN, FP, FN = results

        print("--- Confusion Matrix ---\n")
        print(f"Model: {modelName}\n")
        print(f"TP:{TP}, TN:{TN}, FP:{FP}, FN:{FN}\n")
        
        if TP + FP + FN + TN > 0:
            accuracy = (TP + TN) / (TP + FP + FN + TN)
            print(f"Accuracy: {accuracy:.4f}\n")
        else:
            print("Accuracy: N/A\n")
        
        if TP + FP > 0:
            precision = TP / (TP + FP)
            print(f"Precision: {precision:.4f}\n")
        else:
            print("Precision: N/A (no positive predictions)\n")

        if TP + FN > 0:
            recall = TP / (TP + FN)
            print(f"Recall: {recall:.4f}\n")
        else:
            print("Recall: N/A (no actual positive samples)\n")

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"F-score: {f1:.4f}\n")
        else:
            print("F-score: N/A (precision and recall are both zero)\n")
        return 
    
    def average_model_states(model_states):
        """
        Average the states of multiple models.
        Args:
        model_states (list): A list of model state dictionaries.
        Returns:
        dict: A state dictionary containing the averaged model parameters.
        """
        avg_state = {}
        
        # Get the keys (layer names) from the first state dict
        keys = model_states[0].keys()
        
        for key in keys:
            # Stack all tensors for this key across models
            stacked = torch.stack([state[key] for state in model_states])
            
            # Convert to float before computing mean
            stacked = stacked.float()
            
            # Compute the mean along the first dimension (across models)
            avg_state[key] = torch.mean(stacked, dim=0)
        
        return avg_state
    
    # Save the states of the trained model
    all_model_states = []
    
    # # iterate through each fold from "trainning dataset"
    # for fold, train_loader, val_loader in stratified_KFold_loader(train_data, batchSize=batchSize):
    #     #  ---  analyze dataloader  ---  #
    #     print(f' -- Fold{fold+1} -- ')
    #     print('Train_loader: ')
    #     analyze_dataloader(train_loader)
    #     print('Validate_loader: ')
    #     analyze_dataloader(val_loader)
    #     #  ----------------------------  #
        
    #     # Define the loss function and optimizer 
    #     loss_fn = nn.CrossEntropyLoss(weight = ratio_of_labels(train_loader).to('mps'))
    #     optimizer = optim.SGD(cnnModel.parameters(), lr=0.001, momentum=0.9)
        
    #     # train the model --->. train_loader
    #     cnnModel = getModel(cnn_name) # create a new model instance for each fold 
    #     trainedModel = trainModel(num_epochs, train_loader, device, cnnModel, optimizer, loss_fn)
        
    #     # Save the dictionary containing the state of the model 
    #     all_model_states.append(trainedModel.state_dict())
        
    #     # validate the model ---> val_loader
    #     fp, fn, tn, tp, results= validationModel(val_loader, device, trainedModel)
        
    #     PrintConfusionMatrix(results, cnn_name)
    #     print()
    
    # # Save the avg model state
    # if args.tag is None:
    #     modelName = f'{cnn_name}_e{num_epochs}'
    # else: 
    #     modelName = f'{cnn_name}_e{num_epochs}_{args.tag}'
    # # Caculate the average state of the modelss
    # avg_model_state = average_model_states(all_model_states)
    # torch.save(avg_model_state, f'models/{modelName}.model')
    
    
    
    # Testing my code run sucessfully (no stratified kFold)
    # dataset -> dataloader 
    train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
    # Define the loss function and optimizer 
    loss_fn = nn.CrossEntropyLoss(weight = ratio_of_labels(train_loader).to('mps'))
    optimizer = optim.SGD(cnnModel.parameters(), lr=0.001, momentum=0.9)
    trainedModel = trainModel(num_epochs, train_loader, device, cnnModel, optimizer, loss_fn)
    modelName = f'{cnn_name}_e{num_epochs}_noFold'
    torch.save(trainedModel.state_dict(), f'models/{modelName}.model')
    


if __name__ == "__main__":        
    algorithm()