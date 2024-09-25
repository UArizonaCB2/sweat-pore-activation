import os, argparse, torch, cv2
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset 
from CNNs import SimpleCNN_p32, SimpleCNN_p17, CNN4Layers_p32, CNN4LayersV2_p32, CNN4LayersV3_p32, CNN4LayersV4_p32
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--CNNmodel',
                    required=True,
                    type = str)

parser.add_argument('--prediction',
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
    predictedImg = args.prediction
    
    def recreate_model_architecture(cnn_name):
        # Recreate the model architecture for loading cnn models
        architecture = cnn_name.split("_e")[0]
        
        cnn_models = {
        "SimpleCNN_p32": SimpleCNN_p32.SimpleCNN_p32(),
        "SimpleCNN_p17": SimpleCNN_p17.SimpleCNN_p17(), 
        "CNN4Layers_p32": CNN4Layers_p32.CNN4Layers_p32(),
        "CNN4LayersV2_p32": CNN4LayersV2_p32.CNN4LayersV2_p32(),
        "CNN4LayersV3_p32": CNN4LayersV3_p32.CNN4LayersV3_p32(),
        "CNN4LayersV4_p32": CNN4LayersV4_p32.CNN4LayersV4_p32()}
        
        if architecture in cnn_models:
            cnnModel = cnn_models[architecture]
        else:
            raise ValueError(f"Unsupported CNN Model: {cnn_name}")
        return cnnModel
    
    def create_Dataloaders(dataset_path, train_indices_path, test_indices_path, batch_size=batchSize):
        """
        Get the dataloaders: total_loader = train_loader(80%) + test_loader(20%)
        """
        dataset = torch.load(dataset_path)
        train_indices = torch.load(train_indices_path)
        test_indices = torch.load(test_indices_path)
        # print(f'Total data: {len(dataset)}')
        # print(f'Total training data: {len(train_indices)}')
        # print(f'Total validating data: {len(test_indices)}')
        
        trainSubset = Subset(dataset, train_indices)
        validateSubset = Subset(dataset, test_indices)
        
        total_loader = DataLoader(dataset, batch_size, shuffle=True)
        train_loader = DataLoader(trainSubset, batch_size, shuffle=True)
        validate_loader = DataLoader(validateSubset, batch_size, shuffle=True)
        return total_loader, train_loader, validate_loader

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
        
        label0Length = 0
        label1Length = 0
        
        # disable gradient calculation
        with torch.no_grad():
            for images, labels, names in test_loader:
                # Count the labels
                for label in labels: 
                    if label == 0:
                        label0Length+=1
                    else:
                        label1Length+=1
                
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
        print(f'Patches with no Pores [label0]: {label0Length}')
        print(f'Patches have Pores [label1]: {label1Length}')
        print()
        return fp_names, fn_names, tn_names, tp_names, [TP, TN, FP, FN]
    
    def ConfusionMatrix(results, modelName, predictedImg):
        TP, TN, FP, FN = results

        print("--- Confusion Matrix ---\n")
        print(f"Model: {modelName}\n")
        print(f'Evaluation on: {predictedImg}\n')
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
            
        if TN + FP > 0:
            spec = TN / (TN + FP)
            print(f"Specificity: {spec:.4f}\n")
        else: 
            print("Specificity: N/A (no positive predictions)\n")

        if TP + FN > 0:
            recall = TP / (TP + FN)
            print(f"Sensitivity: {recall:.4f}\n")
        else:
            print("Sensitivity: N/A (no actual positive samples)\n")

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            print(f"F-score: {f1:.4f}\n")
        else:
            print("F-score: N/A (precision and recall are both zero)\n")
        return 
    
    def heatMap(initImgDir, predictedImg, patchSize, fp, fn, tn, tp):
        """
        Args: Original img
              Confusion matrix: fp, fn, tp, tn (name of the images)
        
        This function will draw different colors on all the patches
        according to the confusionMatrix on top of the original image 
        so that human can identify the result from the experiment
        """
        
        # extract the patch number from fp, fn, tp and tn
        # stores only numbers in each list correspondingly 
        fp_patchesNum, fn_patchesNum, tp_patchesNum, tn_patchesNum = [], [], [], []
        matrix_numList = [fp_patchesNum, fn_patchesNum, tp_patchesNum, tn_patchesNum]
        matrix = [fp, fn, tp, tn]
        for i in range(4):
            for patch in matrix[i]:
                patchNum = int(patch.split('_')[1])
                matrix_numList[i].append(patchNum)
                
        # print(f'fp patches: {len(matrix_numList[0])}')
        # print(f'fn patches: {len(matrix_numList[1])}')
        # print(f'tp patches: {len(matrix_numList[2])}')
        # print(f'tn patches: {len(matrix_numList[3])}')
        
        initImg = cv2.imread(os.path.join(initImgDir, f"{predictedImg[0]}.bmp"))
    
        # print(f'(height, width, channels): {initImg.shape}')
        # print(f'patch size: {patchSize}')
        
        height, width = initImg.shape[0], initImg.shape[1]
        
        num_of_x_patches = width // patchSize
        num_of_y_patches = height // patchSize
        
        # Define colors for each category (in BGR format)
        fp_color = (0, 0, 255)  # Red
        fn_color = (255, 0, 0)  # Blue
        tp_color = (0, 255, 0)  # Green
        # tn_color will not be fill up any color

        currentPatchNum = 1 #patch number start with 1
        for y in range(0, num_of_y_patches):
            for x in range(0, num_of_x_patches):
                # Calculate the coordinates of the current patch
                x1 = x * patchSize
                y1 = y * patchSize
                x2 = x1 + patchSize
                y2 = y1 + patchSize
                
                # Check which category the current patch belongs to
                if currentPatchNum in fp_patchesNum:
                    color = fp_color
                elif currentPatchNum in fn_patchesNum:
                    color = fn_color
                elif currentPatchNum in tp_patchesNum:
                    color = tp_color
                else:
                    color = None
                
                # Draw a transparent overlay
                overlay = initImg.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                alpha = 0.3  # Transparency factor
                cv2.addWeighted(overlay, alpha, initImg, 1 - alpha, 0, initImg)
                
                currentPatchNum+=1
                
                
        # print(currentPatchNum)
        
        initImg_rgb = cv2.cvtColor(initImg, cv2.COLOR_BGR2RGB)
        # Display the image
        plt.imshow(initImg_rgb)
        plt.title(f"Image: {predictedImg}")
        plt.axis('off')  # Hide axes
        plt.show()
        
        return 
    
    # Load the state dict
    model_path = f'models/{cnn_name}'
    state_dict = torch.load(model_path)

    trainedModel = recreate_model_architecture(cnn_name)
    
    # Load the state dict into the model
    trainedModel.load_state_dict(state_dict)
    
    # set the model to evaluation mode 
    trainedModel.eval()
    
    total_loader, train_loader, test_loader = create_Dataloaders(
    f'Preprocessing/dataset/{patchSize}X{patchSize}/dataset.pt',
    f'Preprocessing/dataset/{patchSize}X{patchSize}/train_indices.pt',
    f'Preprocessing/dataset/{patchSize}X{patchSize}/test_indices.pt')
    
    fp, fn, tn, tp, results= evaluateModel(total_loader, device, trainedModel)
    
    ConfusionMatrix(results, cnn_name, predictedImg)
    
    initImgDir = f'Preprocessing/input_images/testingModel/{predictedImg}/annotated'
    heatMap(initImgDir,  predictedImg, patchSize, fp, fn, tn, tp)
    

if __name__ == "__main__":
    algorithm()