import os, argparse, torch
from torch.utils.data import Dataset, DataLoader, Subset
from CNNs import SimpleCNN_p32, SimpleCNN_p17, CNN4Layers_p32
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image

parser = argparse.ArgumentParser()

parser.add_argument('--CNNmodel',
                    required=True,
                    type = str)

parser.add_argument('--device',
                    default="mps",
                    type=str)

parser.add_argument('--patchSize',
                    default=32,
                    type=int)

parser.add_argument('--batchSize',
                    default=8,
                    type=int)

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
    This class is going to run the pre-trained model 
    It requires: device, pre-trained model, and test_loader
    It evaluates the confusion matrix and saves imgs from TP, TN FP and FN
    """
    # get the value from hyper parameter
    cnn_name = args.CNNmodel
    device = args.device
    patchSize = int(cnn_name.split("_p")[1].split("_")[0])
    batchSize = args.batchSize
    
    def recreate_model_architecture(cnn_name):
        # Recreate the model architecture for loading cnn models
        architecture = cnn_name.split("_e")[0]
        
        cnn_models = {
        "SimpleCNN_p32": SimpleCNN_p32.SimpleCNN_p32(),
        "SimpleCNN_p17": SimpleCNN_p17.SimpleCNN_p17(), 
        "CNN4Layers_p32": CNN4Layers_p32.CNN4Layers_p32()}
        
        if architecture in cnn_models:
            cnnModel = cnn_models[architecture]
        else:
            raise ValueError(f"Unsupported CNN Model: {cnn_name}")
        return cnnModel
    
    # Load the state dict
    model_path = f'models/{cnn_name}'
    state_dict = torch.load(model_path)

    trainedModel = recreate_model_architecture(cnn_name)
    
    # Load the state dict into the model
    trainedModel.load_state_dict(state_dict)
    
    # set the model to evaluation mode 
    trainedModel.eval()

    # Load the testing dataset
    test_data = torch.load('Preprocessing/dataset/test_indices.pt')
    print(f'Total testing data: {len(test_data)}')
    print()
    
    # Split the data -- Train Validate Test
    test_loader = DataLoader(test_data, batch_size = batchSize, shuffle = True, num_workers = 0)

    def evaluateModel(test_loader, device, model):
        model = model.to(device)
        
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
                        # Create the full path for the image file
                        save_dir = f'results/TP/'
                        filename = names[i]
                        full_path = os.path.join(save_dir, filename)
                        # Save the image
                        save_image(images[0], full_path)
                    
                        tp_names.append(names[i])
                        TP += 1
                    elif predicted_classes[i].item() == labels[i].item() == 0:
                        # Create the full path for the image file
                        save_dir = f'results/TN/'
                        filename = names[i]
                        full_path = os.path.join(save_dir, filename)
                        # Save the image
                        save_image(images[0], full_path)
                        tn_names.append(names[i])
                        TN += 1
                    elif predicted_classes[i].item() != labels[i].item() and predicted_classes[i].item() == 1:
                        # Create the full path for the image file
                        save_dir = f'results/FP/'
                        filename = names[i]
                        full_path = os.path.join(save_dir, filename)
                        # Save the image
                        save_image(images[0], full_path)
                        fp_names.append(names[i]) # Keep track of the fp
                        FP += 1
                    else:
                        # Create the full path for the image file
                        save_dir = f'results/FN/'
                        filename = names[i]
                        full_path = os.path.join(save_dir, filename)
                        # Save the image
                        save_image(images[0], full_path)
                        fn_names.append(names[i]) # Keep track of the fn
                        FN += 1
                        
        confusionMatric["TP: "] = TP
        confusionMatric["FP: "] = FP
        confusionMatric["TN: "] = TN
        confusionMatric["FN: "] = FN
                
        return fp_names, fn_names, tn_names, tp_names, [TP, TN, FP, FN]
    
    fp, fn, tn, tp, results= evaluateModel(test_loader, device, trainedModel)
    
    print(f'False Negatives: {fn}')
    
    def ConfusionMatrix(results, modelName):
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
    
    ConfusionMatrix(results, cnn_name)
    
    

if __name__ == "__main__":
    algorithm()