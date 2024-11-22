import os, argparse, torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset 
from CNNs import SimpleCNN_p32, SimpleCNN_p17, CNN4Layers_p32, CNN4LayersV2_p32, CNN4LayersV3_p32, CNN4LayersV4_p32
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from Preprocessing.centralize_sweatpores import centralize
from scipy.stats import gaussian_kde

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

parser.add_argument('--patchesDir',
                    required=True,
                    type = str)

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
    patches_dir = args.patchesDir
    
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
    
    def create_EvaluateLoaders(dataset_path, batch_size=batchSize):
        """
        This function create data loaders for centralized pores dataset 
        """
        trans = transforms.Compose([
        transforms.Grayscale(),  # Convert RGB to grayscale 3 ---> 1 
        transforms.ToTensor(), # Convert data to Tensor for dataloader
        transforms.Normalize(mean=0.5, std=0.5) # scale the pixel values of the image between -1 and 1
        ])
        
        dataset = SweatPoresDataset(dataset_path, transforms=trans)
        
        eval_loader = DataLoader(dataset, batch_size, shuffle=False)

        return eval_loader

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
        
        # This hashtable will return a dictionary 
        # with the key of string and a list of tuples (name, probability) as value
        specificity = {
            "TP": [],
            "FP": []
        }

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        label0Length = 0
        label1Length = 0
        
        # Opening and writing to a text file
        log_path = os.path.join('results/', 'evaluation_results.txt')
        with open(log_path, 'w') as log_file:
            # 'w' mode: Opens for writing, creating a new file or overwriting existing file
            # Write header
            log_file.write("=== Model Evaluation Results ===\n\n")
            
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

                    # distribution with x-axis (thresholds) / y-axis (sencitivity and specificity)
                    # Finds the highest probabilities for each image
                    probs, predicted_classes = torch.max(probabilities, 1)
                    
                    # Apply threshold , 0.8 would be a good threshold for our dataset
                    # torch.where(condition, input (if true), other (if false))
                    probability_threshold = 0.5
                    predicted_classes = torch.where(probs >= probability_threshold, 
                              predicted_classes, 
                              torch.zeros_like(predicted_classes))

                    for i in range(len(labels)):
                        # Get probability for the predicted class
                        prob = probs[i].item()
                        name = names[i]
                        predicted_label = predicted_classes[i].item()
                        actual_label = labels[i].item()
                        
                        
                        # Format the prediction result
                        result_str = (f"Image: {name} | "
                                    f"Predicted: {predicted_label} | "
                                    f"Actual: {actual_label} | "
                                    f"Probability: {prob:.2%}\n")
                                    # formats a number as a percentage with 2 decimal places
                        # Write to log file
                        log_file.write(result_str)
                        
                        if predicted_classes[i].item() == labels[i].item() == 1:
                            # # Create the full path for the image file
                            # save_dir = f'results/TP/'
                            # filename = names[i]
                            # full_path = os.path.join(save_dir, filename)
                            # # Save the image
                            # save_image(images[0], full_path)
                            tp_names.append(names[i])
                            specificity["TP"].append((name, predicted_label, actual_label, prob))
                            TP += 1
                        elif predicted_classes[i].item() == labels[i].item() == 0:
                            # # Create the full path for the image file
                            # save_dir = f'results/TN/'
                            # filename = names[i]
                            # full_path = os.path.join(save_dir, filename)
                            # # Save the image
                            # save_image(images[0], full_path)
                            tn_names.append(names[i])
                            TN += 1
                        elif predicted_classes[i].item() != labels[i].item() and predicted_classes[i].item() == 1:
                            # # Create the full path for the image file
                            # save_dir = f'results/FP/'
                            # filename = names[i]
                            # full_path = os.path.join(save_dir, filename)
                            # # Save the image
                            # save_image(images[0], full_path)
                            fp_names.append(names[i]) # Keep track of the fp
                            specificity["FP"].append((name, predicted_label, actual_label, prob))
                            FP += 1
                        else:
                            # # Create the full path for the image file
                            # save_dir = f'results/FN/'
                            # filename = names[i]
                            # full_path = os.path.join(save_dir, filename)
                            # # Save the image
                            # save_image(images[0], full_path)
                            fn_names.append(names[i]) # Keep track of the fn
                            FN += 1
                        
        confusionMatric["TP: "] = TP
        confusionMatric["FP: "] = FP
        confusionMatric["TN: "] = TN
        confusionMatric["FN: "] = FN
        print(f'Patches with no Pores [label0]: {label0Length}')
        print(f'Patches have Pores [label1]: {label1Length}')
        print()
        return fp_names, fn_names, tn_names, tp_names, [TP, TN, FP, FN], specificity
    
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
        matrix_numList = [fp_patchesNum, fn_patchesNum, tp_patchesNum, tn_patchesNum] # the patches are lined up ordinarily
        fp_patchesCoord, fn_patchesCoord, tp_patchesCoord, tn_patchesCoord = [], [], [], []
        matrix_coordList = [fp_patchesCoord, fn_patchesCoord, tp_patchesCoord, tn_patchesCoord] # patches are named by coordinates
        matrix = [fp, fn, tp, tn]
        for i in range(4):
            for patch in matrix[i]: # Naming 1: Coordinates
                if 'x' and 'Y' in patch:
                    # print('coordinate naming system')
                    getCoord = patch.split('_')[1].split('X')
                    coordStr = getCoord[1].split('Y')
                    x = int(coordStr[0])
                    y = int(coordStr[1])
                    # print(patch, x, y) # Make sure we extract the coord correctly
                    matrix_coordList[i].append((x,y))
                else: # Naming 2: Idx
                    # print('indx naming system')
                    patchNum = int(patch.split('_')[1])
                    matrix_numList[i].append(patchNum)
        
        # print(matrix_numList)
        # print(f'fp patches: {len(matrix_numList[0])}')
        # print(f'fn patches: {len(matrix_numList[1])}')
        # print(f'tp patches: {len(matrix_numList[2])}')
        # print(f'tn patches: {len(matrix_numList[3])}')
        
        # print(matrix_coordList)
        # print(f'fp patches: {len(matrix_coordList[0])}')
        # print(f'fn patches: {len(matrix_coordList[1])}')
        # print(f'tp patches: {len(matrix_coordList[2])}')
        # print(f'tn patches: {len(matrix_coordList[3])}')
        
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
        
        # Handle which naming system we are using
        coordList, numList = False, False
        if matrix_coordList != [[], [], [], []]:
            print("Coordinates Naming System")
            coordList = True
        elif matrix_numList != [[], [], [], []]:
            print("Index Naming System")
            numList = True
        
        if coordList:
            for idx, list in enumerate(matrix_coordList):
                # print(idx, list)
                # Assign color map 
                if idx == 0: 
                    color = fp_color
                elif idx == 1:
                    color = fn_color
                elif idx == 2:
                    color = tp_color
                else:
                    color = None
                
                # Create patches 
                if list != []:
                    for coord in list:
                        x1 = coord[0] - patchSize//2
                        y1 = coord[1] - patchSize//2
                        x2 = coord[0] + patchSize//2
                        y2 = coord[1] + patchSize//2
            
                        # Draw a transparent overlay ------------------------------- #
                        overlay = initImg.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        alpha = 0.3  # Transparency factor
                        cv2.addWeighted(overlay, alpha, initImg, 1 - alpha, 0, initImg)
                        # ---------------------------------------------------------- #
        elif numList:
            currentPatchNum = 1 #patch number start with 1
            for y in range(0, num_of_y_patches):
                for x in range(0, num_of_x_patches):
                    # Calculate the coordinates of the current patch
                    # Top left corner 
                    x1 = x * patchSize
                    y1 = y * patchSize
                    # Bottom right corner 
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
                    
                    # Draw a transparent overlay ------------------------------- #
                    overlay = initImg.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    alpha = 0.3  # Transparency factor
                    cv2.addWeighted(overlay, alpha, initImg, 1 - alpha, 0, initImg)
                    # ---------------------------------------------------------- #
                    
                    currentPatchNum+=1 
            # print(currentPatchNum)
        
        initImg_rgb = cv2.cvtColor(initImg, cv2.COLOR_BGR2RGB)
        # Display the image
        plt.imshow(initImg_rgb)
        plt.title(f"Image: {predictedImg}")
        plt.axis('off')  # Hide axes
        plt.show()
        
        return 
    
    def specificity_distribution(specificity):
        """
        Creates a distribution with probability on x-axis and PDF of TP and FP on y-axis
        """
        # print(f"Specificity: {specificity}")

        tp_probs = [item[3] for item in specificity["TP"]]
        fp_probs = [item[3] for item in specificity["FP"]]
        
        # print(f"Number of TP: {len(tp_probs)}, Number of FP: {len(fp_probs)}")

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot histograms
        # alpha controls the transparancy
        plt.hist(tp_probs, bins=20, alpha=0.7, label='True Positives', color='blue', density=True)
        plt.hist(fp_probs, bins=20, alpha=0.7, label='False Positives', color='red', density=True)

        # Customize the plot
        plt.xlabel('Probability')
        plt.ylabel('Normalized Density')
        plt.title('Normalized Histogram of True Positives and False Positives')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Show the plot
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
    
    # Create dataloader for evaluation
    eval_loader = create_EvaluateLoaders(patches_dir)
    fp, fn, tn, tp, results, specificity = evaluateModel(eval_loader, device, trainedModel)
    
    # specificity_distribution(specificity)
    
    ConfusionMatrix(results, cnn_name, predictedImg)
    initImgDir = f'Preprocessing/input_images/testingModel/{predictedImg}/raw'
    # heatMap(initImgDir,  predictedImg, patchSize, fp, fn, tn, tp)
    

if __name__ == "__main__":
    algorithm()