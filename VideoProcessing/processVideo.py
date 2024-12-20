import os
import subprocess

def extract_frames(input_video, output_dir):
    # Construct the FFmpeg command
    command = [
        'ffmpeg',
        '-i', input_video,
        '-vf', 'fps=fps=1',  # Extract 1 frame per second, adjust as needed
        os.path.join(output_dir, 'frame%04d.bmp')
    ]

    # Execute the FFmpeg command
    subprocess.run(command, check=True)
    return

# Define paths based on the given directory structure
input_video = './videos/inputVideos/hand.mp4'
output_frames = './frames/videoToframes'

# Extract frames
extract_frames(input_video, output_frames)