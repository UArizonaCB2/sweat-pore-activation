import cv2
import imutils
import argparse
import os
import shutil
import numpy as np
from skimage.util import random_noise
import sys

class Preprocessing:
    def __init__(self):
        pass

    def process_image(self, raw_image_path, circled_image_path, image_name):
        """
        Function to load Dr. Runyon's annotated image with circled sweat pores, isolates the sweat pores based on the 
        circled regions, computes the centroid of the sweat pores, and performs data augmentation by adding noise to 
        the raw image.
        Inputs: 
            raw_image_path - Path of raw/original sweat pore image without any annotations
            circled_image_path - Path of Dr Runyon's annotated image
            image_name - The name of the current processing image
        """
        # Load Dr Runyon's image with circled sweat pores
        circled_image = cv2.imread(circled_image_path)

        # Isolate the sweat pores based on the circled regions of the image
        contour_image = self.isolate_sweat_pores(circled_image)

        # Compute centroid of sweat pores
        centroid_coordinates = self.centroid(contour_image, image_name)

        # Perform Data Augmentation (Add Noise)
        # self.data_augmentation(raw_image_path, centroid_coordinates)

    def isolate_sweat_pores(self, circled_image):
        """
        Function to isolate circled regions of sweat pores from Dr Runyon's annotated images
        Inputs:
            circled_image - Path of Dr Runyon's annotated image
        Output:
            contour_image - Isolated regions of sweat pores with black mask as background
        """
        # Make sure the image is loaded
        if circled_image is None:
            print("Failed to load the image.")

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(circled_image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for green, yellow, and red color in HSV (Adjust these values as needed)
        # Green circles
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([90, 245, 245]) 
        # Red circles
        lower_red = np.array([0, 100, 100])  
        upper_red = np.array([20, 255, 255]) 
        # Yellow circles
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255]) 

        # Threshold the HSV image to isolate green, red, and yellow regions
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine the red green, and yellow mask into just one mask
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask, red_mask), yellow_mask)
        #combined_mask = cv2.bitwise_or(green_mask, red_mask)

        # Find contours of the green, red, and yellow regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask for background
        blank_mask = np.zeros_like(circled_image[:, :, 0]) # Black mask
        #blank_mask = np.full_like(image[:, :, 0], 255) # White mask

        # Draw contours on the blank mask
        cv2.drawContours(blank_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the original image
        contour_image = cv2.bitwise_and(circled_image, circled_image, mask=blank_mask)

        # Save or display the result
        #cv2.imwrite("isolated_RGYcircles.png", result)
        
        return contour_image

    def centroid(self, contour_image, image_name):
        """
        Function to compute the centroid of the isolated sweat pore images
        Input:
            contour_image - Isolated regions of sweat pores with black mask as background
            image_name - The name of the current processing image name
            coordinate_filename - Path of the file where centroids coordinates are stored.
            centroid_filename - Path of the file which has the image with circled centroids.
        Output:
            coordinate_filename - Centroid coordinate filename
        """
         # Generate unique filenames based on the input image filename
        coordinate_filename = f"{image_name}.txt"
        centroid_filename = f"{image_name}.png"
        
        # https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
        # For non greyscaled images (circled images)
        # Load the image, convert it to grayscale, blur it slightly, and threshold it
        image = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1] # Intensity upper threshold = 60

        # For Grey scaled image
        # Load the grayscale filtered image
        # grey = cv2.imread(self.isolated_sp, cv2.IMREAD_GRAYSCALE)
        # thresh = cv2.threshold(grey, 60, 255, cv2.THRESH_BINARY)[1] 

        # find contours in the thresholded image
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the contour image to draw centroids on
        contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # -- Ricky Modified -- #
        # Specify the directory where you want to save the image
        coordinate_output_directory = "../results/centroid_coordinates/"
        
        # Create the full path for the centroid image file
        coordinate_filepath = os.path.join(coordinate_output_directory, coordinate_filename)
        # -------------------- #
        
        # For creating GroundTruth files
        with open(coordinate_filepath, "w") as file:
            # Write the header
            file.write("y\tx\n")  

            # Loop over the contours
            for c in cnts:
                # Compute the area of the contour
                area = cv2.contourArea(c)
                
                # Skip processing if the contour area is zero
                if (area == 0):
                    continue

                # Compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Write the coordinates to the file
                file.write(f"{cY}\t{cX}\n")  # For some reason, the PoreGroundTruthMarked images must be in (Y X) format instead of (X Y)  

                # Draw the contour and centroid on the image for visualization
                cv2.drawContours(contour_image, [c], -1, (255, 0, 0), 1) # Blue Contour
                cv2.circle(contour_image, (cX, cY), 0, (0, 0, 255), -1) # Red Centroid
        
        # -- Ricky Modified -- #
        # Specify the directory where you want to save the image
        centroid_output_directory = "../results/contour_images/"

        # Create the full path for the centroid image file
        centroid_filepath = os.path.join(centroid_output_directory, centroid_filename)
        # Save contour and centroid image in the specified directory
        cv2.imwrite(centroid_filepath, contour_image)
        # -------------------- #

        return coordinate_filename

    def data_augmentation(self, raw_image_path, centroid_coordinates, output_folder='../dataset/'):
        """
        Function to perform data augmentation (Add noise) on raw/original image and pair it with the centroid coordinates calculated.
        Inputs:
            raw_image_path - Raw/original sweat pore image without any annotations
            centroid_coordinates - Centroid coordinate filename
            output_folder - Path to where the augmentation images and co-ordinates will be stored.
        Output:
            Function populates augmented images and corresponding centroid coordinate files in their directories
        """
        # Directory where the images will be saved
        output_image_directory = os.path.join(output_folder, 'PoreGroundTruthSampleimage')
        #output_image_directory = f"{output_folder}/PoreGroundTruthSampleimage/"
        output_txt_directory = os.path.join(output_folder, 'PoreGroundTruthMarked')
        #output_txt_directory = "dataset/PoreGroundTruthMarked/"

        # Get the list of files in the output directory
        existing_files = os.listdir(output_image_directory)

        # Get the number of existing files
        num_existing_files = len(existing_files)

        # Read original image
        original_image = cv2.imread(raw_image_path, 1)

        # Define noise levels between 0 and 0.1
        noise_levels = [0.01, 0.03, 0.05, 0.07, 0.09]

        # Iterate over noise levels
        for noise_level in noise_levels:
            # For reproducibility 
            np.random.seed(42)  
            
            # Add Gaussian noise to original image
            noisy_image = random_noise(original_image, mode='gaussian', var=noise_level, clip=True)
            
            # Convert noisy image to uint8 (necessary for saving as image)
            noisy_image_uint8 = (255 * noisy_image).astype(np.uint8)
            
            # Determine the next available index
            next_index = num_existing_files + 1

            # Save the image with the next available index
            noisy_image_filename = os.path.join(output_image_directory, f'{next_index}.bmp')
            cv2.imwrite(noisy_image_filename, noisy_image_uint8)

            # Create corresponding txt file for centroid coordinates
            txt_filename = os.path.join(output_txt_directory, f'{next_index}.txt')
            
            # Copy centroid_coordinates file and just rename it
            shutil.copyfile(centroid_coordinates, txt_filename)
            
            # Increment the count of existing files
            num_existing_files += 1  
            

if __name__ == "__main__":
    # create an instance of the Procesing class
    processor = Preprocessing()
    
    # check the user's inputs
    if len(sys.argv) != 3:
        print("Pass the raw and circled image folder paths as arguments.")
        exit()
        
    # Define paths for the folders
    raw_image_folder = sys.argv[1]
    circled_image_folder = sys.argv[2]
    
    # Get a list of image files in the raw image folder
    raw_image_files = os.listdir(raw_image_folder)

    for image_name in raw_image_files:
        """
        We assume that the circled_image_files have the same name as
        their correspoding raw_image_files
        """
        # construct the full path of the raw image and circled image paths
        raw_image_path = os.path.join(raw_image_folder, image_name)
        circled_image_path = os.path.join(circled_image_folder, image_name)
        
        print(raw_image_path)
        print(circled_image_path)
        
        # Run the Preprocessing pipeline
        processor.process_image(raw_image_path, circled_image_path, image_name)
        
    # Print the number of processed image pairs
    print(f"Processed {len(raw_image_files)} image pairs.")
        
    
    
    
    
    
