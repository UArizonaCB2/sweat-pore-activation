import os, argparse, torch
from CNNs import SimpleCNN

parser = argparse.ArgumentParser()

parser.add_argument('--CNNmodel',
                    required=True,
                    type = str)

args = parser.parse_args()

class algorithm:
    # get the value from hyper parameter
    cnn_name = args.CNNmodel
    
    def recreate_model_architecture(cnn_name):
        # Recreate the model architecture for loading cnn models
        architecture = cnn_name.split("_")[0]
        
        cnn_models = {
        "SimpleCNN": SimpleCNN.SimpleCNN()}
        
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

    print("Set the trainedModel to evaluation mode")
    
    
    
    
        

if __name__ == "__main__":
    algorithm()