import os, argparse, torch, cv2, sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from CNNs import CNN4Layers_p32

parser = argparse.ArgumentParser()

parser.add_argument('--CNNmodel',
                    required=True,
                    type = str)

parser.add_argument('--device',
                    default="mps",
                    type=str)

parser.add_argument('--patchesDir',
                    required=True,
                    type = str)

parser.add_argument('--batchSize',
                    default=8,
                    type=int)

parser.add_argument('--currentFrameNumber',
                    required=True,
                    type = str)

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
        img_name = self.img_files[idx] 

        if self.transform: 
            image = self.transform(image)

        return image, img_name
    
    
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
    patches_dir = args.patchesDir
    frame_num = args.currentFrameNumber
    
    def recreate_model_architecture(cnn_name):
        # Recreate the model architecture for loading cnn models
        architecture = cnn_name.split("_e")[0]
        
        cnn_models = {
        "CNN4Layers_p32": CNN4Layers_p32.CNN4Layers_p32()}
        
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
        # Move counter initialization outside the loops
        label0Count = 0 
        label1Count = 0
        hasPores = []
        # Opening and writing to a text file
        log_path = os.path.join('VideoProcessing/', 'prediction.txt')
        with open(log_path, 'a') as log_file:
            # 'w' mode: Opens for writing, creating a new file or overwriting existing file
            # Write header
            log_file.write("=== Model Prediction Results ===\n\n")
        
            frameName = ""
            
            # disable gradient calculation
            with torch.no_grad():
                for images, names in test_loader:
                    frameName = names[0].split('_')[0]
                    images = images.to(device)
                    
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
                    
                    for i in range(len(names)):
                        # Get probability for the predicted class
                        prob = probs[i].item()
                        name = names[i]
                        predicted_label = predicted_classes[i].item()
                        
                        if predicted_label == 0:
                            label0Count +=1
                        elif predicted_label == 1:
                            label1Count += 1
                            hasPores.append(name)
                        
                        # Format the prediction result
                        result_str = (f"Image: {name} | "
                                    f"Predicted: {predicted_label} | "
                                    f"Probability: {prob:.2%}\n")
                                    # formats a number as a percentage with 2 decimal places
                        # Write to log file
                        log_file.write(result_str)
                        

            infoStr = (f'{frameName} with no Pores [label0]: {label0Count}\n'
                       f'{frameName} have Pores [label1]: {label1Count}\n\n')
            log_file.write(infoStr)
            print(infoStr)
        return hasPores
    
    def createHeatmap(hasPores, storeFrames_path, frame_num):   
        # Extract the index from the patch name
        frame_name = frame_num.split("/")[2] # assume each img has at least one pore detected
        # print(frame_name)
        frame_path = f'VideoProcessing/frames/videoToframes/{frame_name}'
        frame = cv2.imread(frame_path)
        
        # Get the width and height from the frame
        height, width = frame.shape[0], frame.shape[1]
        # floor dividion operation
        num_of_x_patches = width // 32
        num_of_y_patches = height // 32
        # print(frame_name, f"height: {height}", f"height: {width}")
        
        # loop through the hasPores list
        idxList_hasPores = []
        for patch_name in hasPores:
            # Get the index of the patch that is predicted as having pores
            idxList_hasPores.append(int(patch_name.split('_')[1].split('.')[0]))

        
        current_patchIdx = 0
        for y in range(0, num_of_y_patches):
            for x in range(0, num_of_x_patches):
                # Calculate the coordinates of the current patch
                # Top left corner 
                x1 = x * 32
                y1 = y * 32
                # Bottom right corner 
                x2 = x1 + 32
                y2 = y1 + 32
                
                color = (0,255,0) #Green
                
                if current_patchIdx in idxList_hasPores:
                    # Draw a transparent overlay ------------------------------- #
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    alpha = 0.3  # Transparency factor
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    # ---------------------------------------------------------- #
        
                current_patchIdx += 1
                
        # Save the highlighted frame into the directory
        output_path = os.path.join(storeFrames_path, frame_name)
        cv2.imwrite(output_path, frame)
        return
    
    # Load the state dict
    model_path = f'models/{cnn_name}'
    
    # # Check input directory
    # if os.path.exists(model_path):
    #     print("Model Path exist")
    # if os.path.exists(patches_dir):
    #     print("Patches Dir exist")
    
    state_dict = torch.load(model_path)

    trainedModel = recreate_model_architecture(cnn_name)
    
    # Load the state dict into the model
    trainedModel.load_state_dict(state_dict)
    
    # set the model to evaluation mode 
    trainedModel.eval()
    
    # Create dataloader for evaluation
    eval_loader = create_EvaluateLoaders(patches_dir)
    hasPores = evaluateModel(eval_loader, device, trainedModel)
    
    # Recreate frames with detected pores
    storeFrames_path = 'VideoProcessing/frames/framesToVideo/'
    createHeatmap(hasPores=hasPores, storeFrames_path=storeFrames_path, frame_num=frame_num)
    

if __name__ == "__main__":
    algorithm()