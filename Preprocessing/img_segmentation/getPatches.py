import os, cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--patchSize',
                    required=True,
                    type = int)

parser.add_argument('--rawDir',
                    required=True,
                    type = str)

parser.add_argument('--annotatedDir',
                    required=True,
                    type = str)

parser.add_argument('--coordinatesDir',
                    required=True,
                    type = str)

parser.add_argument('--centroidsDir',
                    required=True,
                    type = str)

parser.add_argument('--patchesDir',
                    required=True,
                    type = str)

args = parser.parse_args()

class GetPatches:
    def __init__(self, patch_size = 32):
        self.patch_size = patch_size
        self.centroid_lst = []
    
    def process_images(self, raw_image_path, annotated_image_path, image_name):
        """
        image_name - The name of the current processing image
        This function isolates the sweatpores from the annotated image and 
        computes the centroids and detects if there's any sweat pores in the
        given area for further ML model
        """
        # load the annotated image and raw image 
        annotated_img = cv2.imread(annotated_image_path)
        raw_img = cv2.imread(raw_image_path)
        
        # isolates the sweat pores from the annotated img
        contour_image = self.isolate_sweat_pores(annotated_img)
        
        # Compute centroid of sweat pores
        self.centroid(contour_image, image_name)
        
        # Check if there has sweat pores in the batch
        self.has_sweatPores(raw_img, image_name)
        
        # Empty the centroid lst for the next iteration
        self.centroid_lst = []
        
    def isolate_sweat_pores(self, circled_image):
        """
        Function to isolate circled regions of sweat pores from Dr Runyon's annotated images
        Inputs:
            circled_image - Path of Dr Runyon's annotated image
        Output:
            contour_image - Isolated regions of sweat pores with black mask as background
        """
        # Error Checking
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

        # Find contours of the green, red, and yellow regions
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank mask for background
        blank_mask = np.zeros_like(circled_image[:, :, 0]) # Black mask

        # Draw contours on the blank mask
        cv2.drawContours(blank_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the original image
        contour_image = cv2.bitwise_and(circled_image, circled_image, mask=blank_mask)
        
        return contour_image

    def centroid(self, contour_image, image_name):
        """
        Function to compute the centroid of the isolated sweat pore images
        Function stores both centroid files and coordinate files
        Input:
            contour_image - Isolated regions of sweat pores with black mask as background
            image_name - The name of the current processing image name
            coordinate_filename - Path of the file where centroids coordinates are stored.
            centroid_filename - Path of the file which has the image with circled centroids.
        Output:
            coordinate_filename - Centroid coordinate filename
        """
         # Generate unique filenames based on the input image filename
        image_name = image_name[:-4] # crop the ".bmp" extenstion
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
        coordinate_output_directory = args.coordinatesDir
        
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
                
                # Store the coordinates into a list
                self.centroid_lst.append((cX, cY))

                # Write the coordinates to the file 
                file.write(f"{cY}\t{cX}\n")  # For some reason, the PoreGroundTruthMarked images must be in (Y X) format instead of (X Y)  
                
                # Draw the contour and centroid on the image for visualization
                cv2.drawContours(contour_image, [c], -1, (255, 0, 0), 1) # Blue Contour
                cv2.circle(contour_image, (cX, cY), 0, (0, 0, 255), -1) # Red Centroid
        
        # -- Ricky Modified -- #
        # Specify the directory where you want to save the image
        centroid_output_directory = args.centroidsDir

        # Create the full path for the centroid image file
        contour_filepath = os.path.join(centroid_output_directory, centroid_filename)
        # Save contour and centroid image in the specified directory
        cv2.imwrite(contour_filepath, contour_image)
        # -------------------- #
        return
    
    def has_sweatPores(self, raw_image, image_name):
        """
        This function will store the batches if there are detected sweat pores.
        We created a kernel (32 X 32) to slide through the raw_image. If the coordinates 
        from Coordinates_lst are in the range of the current kernel area than we store and 
        mark the batch. 
        Args:
            raw_image: raw image read by cv.imshow()
            Coordinates_lst: a list of tuples which consist of sweat pores coordinates
        """
        # Get the shape of the inital img
        img_height, img_width, _ = raw_image.shape
        
        # Define the size of each batch
        batch_height = self.patch_size
        batch_width = self.patch_size
        stride = batch_width  # move to the next stride to avoid overlap 
        
        
        # Save the valid batches in this directory
        batches_directory = f"{args.patchesDir}/{self.patch_size}x{self.patch_size}/"
        
        # Check if the directory exists
        if not os.path.exists(batches_directory):
            # If it doesn't exist, create it
            os.makedirs(batches_directory)
            
        
        # Number of batches for the current processing image
        num_batch_count = 0
        hasPores_batches = 0
        pores_count = 0
        label = 0
        for i in range(0, img_height - batch_height + 1, stride):
            for j in range(0, img_width - batch_width + 1, stride):
                # increament num_batch_count
                num_batch_count += 1
                # Extract the current batch
                # Numpy ---> image[height, width, channels]
                batch = raw_image[i:i+batch_height, j:j+batch_width, :]
                
                # Check if there's any sweat pore coordinates within this batch
                has_sweat_pore = False
                for coord in self.centroid_lst:
                    x, y = coord  # Unpack the tuple into x and y coordinates
                    if (j <= x < j+batch_width and i <= y < i+batch_height):
                        # Detect Sweat Pores
                        has_sweat_pore = True
                        pores_count += 1
                        
                # Use the appropriate label based on whether a sweat pore was found
                # label = 1 if has_sweat_pore else 0
                if has_sweat_pore:
                    label = 1
                    hasPores_batches += 1
                else:
                    label = 0
                    
                batch_filename = f"{image_name}_{num_batch_count}_{label}.png"
                batch_path = os.path.join(batches_directory, batch_filename)
                cv2.imwrite(batch_path, batch)
                 
        print("-- Summary --")
        print(f"Image name: {image_name}")
        print("Image Shape:", "(",img_width, img_height,")")
        print("Total Sweat Pore Coordinates: ",len(self.centroid_lst), "| Sweat Pores Count: ",pores_count)
        print("Total Bathces: ", (img_height // batch_height)*(img_width // batch_width),
              "| Batches Count: ",num_batch_count)
        print("Batches have sweat pores: ", hasPores_batches)
        print()
        
        return

if __name__ == "__main__":
    # Get the hyper parameters from runPreprocessing.sh
    patchSize = args.patchSize
    raw_image_folder = args.rawDir
    annotated_image_folder = args.annotatedDir
    
    # create an instance of the getPatches class
    getPatches = GetPatches(patch_size = patchSize)   
    
    # Get a list of image files in the raw image folder
    raw_image_files = os.listdir(raw_image_folder)
    
    DSstoreFileCounter = 0
    for image_name in raw_image_files:
        """
        We assume that the images in the annotated_image_files have the 
        same images name as  the images in their correspoding raw_image_files
        """
        # Skip .DS_Store files
        if image_name.startswith('.'):
            DSstoreFileCounter += 1
            continue
        # construct the full path of the raw image and circled image paths
        raw_image_path = os.path.join(raw_image_folder, image_name)
        annotated_image_path = os.path.join(annotated_image_folder, image_name)
        # # Debug prints
        # print(f"Processing image: {image_name}")
        # print(f"Raw image path: {raw_image_path}")
        # print(f"Raw image exists: {os.path.exists(raw_image_path)}")
        # print(f"Annotated image path: {annotated_image_path}")
        # print(f"Annotated image exists: {os.path.exists(annotated_image_path)}")
        
        # Run the Preprocessing pipeline
        getPatches.process_images(raw_image_path, annotated_image_path, image_name)
        
    # Print the number of processed image pairs
    print(f"Processed {len(raw_image_files) - DSstoreFileCounter} image pairs.\n")