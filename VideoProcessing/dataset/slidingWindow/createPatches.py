import os, cv2, argparse

parser = argparse.ArgumentParser()

parser.add_argument('--frameName',
                    required=True,
                    type = str)

args = parser.parse_args()


class createPatches:
    def __init__(self, name):
        imgName = name
        imgDir = f"./frames/videoToframes" #(called by analyzeSweatPores.sh)
        patchSize = 32
        self.generatePatches(imgDir=imgDir, patchSize = patchSize, imgName=imgName)
        
    def generatePatches(self, imgDir, patchSize, imgName):
        """
        This function gets the img from the imgDir and create patches with given patch size
        Args:
            imgDir : directory of the img 
            patchSize : the size of output patches
        """
        patchlst = []
        
        # Get the frame
        image_path = os.path.join(imgDir, f"{imgName}.bmp")
        frame = cv2.imread(image_path)

        # if os.path.exists(image_path):
        #     print(f"Directory exists")
        # if frame is None:
        #     print(f"Error: Failed to read image at {image_path}")
            
        # Get image dimensions
        height, width, channels = frame.shape
        # print(f'Img shape - height:{height}, width:{width}, channels:{channels}')
        
        #  Define storing directory (called by analyzeSweatPores.sh)
        storingDir = "./dataset/slidingWindow/32X32/"
        
        # Clear out the patches in the storing directory for prior iteration
        for filename in os.listdir(storingDir):
            file_path = os.path.join(storingDir, filename)
            os.remove(file_path)

        # Iterate over the image with stride of 32 (non-overlapping patches)
        idx = 0
        for y in range(0, height - patchSize + 1, patchSize):
            for x in range(0, width - patchSize + 1, patchSize):
                patch = frame[y:y + patchSize, x:x + patchSize]
                patchlst.append(patch)
                # store each pathch into the directory 
                patch_filename = os.path.join(storingDir, f"{imgName}_{idx}.bmp")
                cv2.imwrite(patch_filename, patch)
                idx += 1
        print(f"Generate {len(patchlst)} patches from {imgName}")
        return patchlst
        
if __name__ == "__main__":
    name = args.frameName
    createPatches(name)