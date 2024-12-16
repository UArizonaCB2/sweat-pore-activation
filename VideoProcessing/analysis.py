import re
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import glob

class Analysis:
    def __init__(self):
        print("Analyzing the result...")
        self.data = []
        self.extractData()
        self.scatterplot()
        self.createVideo()
        
    def extractData(self):
        #text = "frame0171 have Pores [label1]: 175"
        # Open and read the file
        with open('prediction.txt', 'r') as file:
            content = file.read()

        pattern = r"frame(\d+) have Pores \[label1\]: (\d+)"

        matches = re.findall(pattern, content)

        for match in matches:
            frame_number = int(match[0])
            pore_count = int(match[1])
            # print(f"Frame Number: {frame_number}")
            # print(f"Pore Count: {pore_count}")
            self.data.append((frame_number, pore_count))
            
        return
    
    def scatterplot(self):
        # Separate the data into x and y lists
        x = [item[0] for item in self.data]
        y = [item[1] for item in self.data]

        # Create the scatter plot
        plt.figure(figsize=(12, 6))
        # plt.scatter(x, y, alpha=0.6)
        
        # Add line connecting points
        plt.plot(x, y, '-', color='lightblue', alpha=0.6)  # Line will be drawn first
        plt.scatter(x, y, alpha=0.5, color='blue')  # Points will be drawn on top

        # Customize the plot
        plt.title('Number of Predicted Pores per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Predicted Pores')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the img
        # Save the plot
        save_path = './results/distribution'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # # Show the plot
        # plt.tight_layout()
        # plt.show()
        return 
    
    def createVideo(self, fps=1):
        # FFmpeg command to create video from frames
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', './frames/framesToVideo/frame*.bmp',  # Updated path to frames
            '-c:v', 'libx264',  # Use H.264 codec
            '-pix_fmt', 'yuv420p',  # Pixel format for better compatibility
            '-preset', 'medium',  # Encoding speed preset
            '-crf', '23',  # Quality setting (lower = better quality, 23 is default)
            './results/output_video.mp4'  # Output file
        ]
        # Execute FFmpeg command
        print("Creating video from frames...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Video creation successful! Output saved as './results/output_video.mp4'")
        else:
            print("Error creating video:")
        
Analysis()