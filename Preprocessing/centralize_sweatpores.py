import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class centralize:
    """
    This class takes the coordinates of sweat pores and create a patch with a given size
    The centroid of the patch has to be the sweat pore coordinate
    Return: a list of patches
    """

    def __init__(self, coord_dir = './testingModel_output_patches/6bmp/centroid_coordinates/', image_dir = './input_images/testingModel/6bmp/raw/'):
        self.patchSize = 32 # Custom your own patch size
        self.batchSize = 8
        self.CoordList = self.getCoordinates(coord_dir)
        self.patchList = self.createPatches(self.patchSize, self.CoordList, image_dir)
        self.storePatches(self.patchList)
        self.visulaizePatch(self.patchList, self.patchSize, patchNumber =45)
        
    def getCoordinates(self, coordDir, filename='6.txt'):
        """
        Get the Sweat pore coordinates from the given directory and return 
        a list of tuple stores coordinates
        """
        coordList = []
        file_path = os.path.join(coordDir, filename)
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.split()[0].isdigit():
                    # Coordinates list stores (x, y), we store the txt file as y x
                    coordList.append((int(line.split()[1]), int(line.split()[0])))
        
        return coordList
    
    def createPatches(self, patchSize, coordList, imgDir):
        """
        Create the patches with a centered sweat pore 
        """
        print(f'PatchSize: ', patchSize)
        print(f'Length of coordList: ',len(coordList))
        
        # Construct the full path to the image
        image_path = os.path.join(imgDir, '6.bmp')
        
        image = cv2.imread(image_path)
        
        # Get the image dimensions
        height, width, channels = image.shape
        
        print(f'Img shape - height:{height}, width:{width}, channels:{channels}')
        
        patchList = []
        
        # create a black blackground for the patch 
        patch = np.zeros((patchSize, patchSize, image.shape[2]), dtype=image.dtype)
        
        half_size = patchSize // 2
        for x,y in coordList:
            left = x - half_size
            right = left + patchSize
            top = y - half_size
            bottom = top + patchSize
            # make sure the patch is in the bound
            if (left >= 0 and right <= width and top >= 0 and bottom <= height):
                # Copy the img within the given region, left, right, top and, bottom
                patch = image[top:bottom, left:right].copy()
                # append a tuple with (patch, Xcoord, Ycoord)
                patchList.append((patch, x, y))
                
        return patchList # patchList is a list of tuples 

    def storePatches(self, patchList):
        """
        This functions store the centralized patches in the local directory 
        for makeing the dataset 
        """
        dir = './centralizedPatches/32X32/'
        for i, (patch, x, y) in enumerate(patchList):
            # Generate a unique filename for each patch
            # filen orignImgName_indx_Xcoord_Ycoord_hasPore.png
            patchName = f"6.bmp_{i}X{x}Y{y}_1.png"
            
            # Construct the full path for the output file
            output_path = os.path.join(dir, patchName)

            # Save the patch as an image file
            cv2.imwrite(output_path, patch)
        
        return

    def visulaizePatch(self, patchList, patchSize, patchNumber):
         # Visualize the a patch
        if patchList:
            patch = patchList[patchNumber][0]
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            plt.title(f"Patch (Size: {patchSize}x{patchSize})")
            plt.axis('off')
            plt.show()
            
            print(f"Shape of first patch: {patch.shape}")
            print(f"Data type of first patch: {patch.dtype}")
        else:
            print("No patches were created. Check your coordinates and image dimensions.")

if __name__ == "__main__":
    centralizer = centralize()
    print(f'Number of pathces: {len(centralizer.patchList)}')
    print(f'Patch Type: {type(centralizer.patchList[0])}')