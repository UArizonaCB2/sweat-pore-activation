import sys, os, cv2

class GetPatches:
    def __init__(self):
        pass
    
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
        coordinates_lst = self.isolates_sweatPores(annotated_img)
        
        # calculate the centoids of the sweat pores
        self.has_sweatPores(raw_img, coordinates_lst)
        
        
    def isolates_sweatPores(self, annotated_img):
        return []
    
    def has_sweatPores(self,raw_img, coordinates_lst):
        return
        

if __name__ == "__main__":
    # create an instance of the getPatches class
    getPatches = GetPatches()
        
    # Define paths for the folders
    raw_image_folder = "../input_images/raw/"
    annotated_image_folder = "../input_images/annotated/"
    
    # Get a list of image files in the raw image folder
    raw_image_files = os.listdir(raw_image_folder)
    
    for image_name in raw_image_files:
        """
        We assume that the images in the annotated_image_files have the 
        same images name as  the images in their correspoding raw_image_files
        """
        # construct the full path of the raw image and circled image paths
        raw_image_path = os.path.join(raw_image_folder, image_name)
        annotated_image_path = os.path.join(annotated_image_folder, image_name)
        
        # Run the Preprocessing pipeline
        getPatches.process_images(raw_image_path, annotated_image_path, image_name)
        
    # Print the number of processed image pairs
    print(f"Processed {len(raw_image_files)} image pairs.")