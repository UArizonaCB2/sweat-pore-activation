
import os
import numpy as np


class centralize:
    """
    This class takes the coordinates of sweat pores and create a patch with a given size
    The centroid of the patch has to be the sweat pore coordinate
    Return: a list of patches
    """
    def __init__(self):
        self.patchSize = 32 # Custom your own patch size
        self.coord_dir = './testingModel_output_patches/2bmp/centroid_coordinates/'# The directory of the coordinates
        self.image_dir = './input_images/testingModel/2bmp/raw' # The directory of the original image
        self.CoordList = self.getCoordinates(self.coord_dir)
        self.patches = self.createPatches(self.patchSize, self.CoordList, self.image_dir)
        
    def getCoordinates(self, coordDir, filename='2.txt'):
        """
        Get the Sweat pore coordinates from the given directory and return 
        a list of tuple stores coordinates
        """
        coordList = []
        file_path = os.path.join(coordDir, filename)
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.split()[0].isdigit():
                    # Coordinates list stores (x, y)
                    coordList.append((int(line.split()[0]), int(line.split()[1])))
        
        return coordList
    
    def createPatches(self, patchSize, coordList, imgDir):
        """
        Create the patches with a centered sweat pore 
        """
        print("This is createPatches function!")
        print(f'PatchSize: ', patchSize)
        print(f'Length of coordList: ',len(coordList))
        
        
        return 
    


if __name__ == "__main__":
    centralize()